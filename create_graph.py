import os
import torch
from torch_geometric.data import InMemoryDataset, HeteroData
from csv import reader # Make sure to import your CSV reader
import pandas as pd
import category_encoders as ce

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def encode_variable_node_indices(numVariableNodes):
    data = pd.DataFrame({'nodeIdx': [i for i in range(numVariableNodes)]})
    encoder = ce.BaseNEncoder(cols=['nodeIdx'], return_df=False, base=2)
    encoded_variable_node_indices = encoder.fit_transform(data)
    
    # Return it directly, as it is already a numpy array
    return encoded_variable_node_indices

def read_from_wls_se_files(datasetType, data_dir):
    # This is your exact legacy function, kept intact.
    # Make sure this function is either defined here or imported.
    if datasetType == "test":
        path = os.path.join(data_dir, "Test_Estimate.csv")
    elif datasetType == "validation":
        path = os.path.join(data_dir, "Validation_Estimate.csv")
    else:
        path = os.path.join(data_dir, "Training_Estimate.csv")
    
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        estimate_rows = list(csv_reader)

    if datasetType == "test":
        path = os.path.join(data_dir, "Test_Jacobian.csv")
    elif datasetType == "validation":
        path = os.path.join(data_dir, "Validation_Jacobian.csv")
    else:
        path = os.path.join(data_dir, "Training_Jacobian.csv")
        
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        jacobian_rows = []
        for i, line in enumerate(csv_reader):
            jacobian_row = [1 if abs(float(element)) > 0.0001 else 0 for element in line]
            jacobian_rows.append(jacobian_row)

    if datasetType == "test":
        path = os.path.join(data_dir, "Test_Measurement.csv")
    elif datasetType == "validation":
        path = os.path.join(data_dir, "Validation_Measurement.csv")
    else:
        path = os.path.join(data_dir, "Training_Measurement.csv")
        
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        measurement_rows = list(csv_reader)

    if datasetType == "test":
        path = os.path.join(data_dir, "Test_Variance.csv")
    elif datasetType == "validation":
        path = os.path.join(data_dir, "Validation_Variance.csv")
    else:
        path = os.path.join(data_dir, "Training_Variance.csv")
        
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        variance_rows = list(csv_reader)

    if datasetType == "test":
        path = os.path.join(data_dir, "Test_Covariance.csv")
    elif datasetType == "validation":
        path = os.path.join(data_dir, "Validation_Covariance.csv")
    else:
        path = os.path.join(data_dir, "Training_Covariance.csv")
        
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        covariance_rows = list(csv_reader)

    if datasetType == "test":
        path = os.path.join(data_dir, "Test_NumMeasurements.txt")
    elif datasetType == "validation":
        path = os.path.join(data_dir, "Validation_NumMeasurements.txt")
    else:
        path = os.path.join(data_dir, "Training_NumMeasurements.txt")
        
    with open(path, 'r') as file1:
        lines = file1.readlines()
        numLinePMeasurements = int(lines[0])
        numPInjMeasurements = int(lines[1])
        numThetaMeasurements = int(lines[2])
        numGraphs = int(lines[3])
        numVariableNodes = int(lines[4]) * 2  # Vream and Vimag
        numMeasurements = numLinePMeasurements + numPInjMeasurements + numThetaMeasurements

    return numGraphs, numVariableNodes, numMeasurements, estimate_rows, jacobian_rows, measurement_rows, variance_rows, covariance_rows

# ==========================================
# 2. DATASET CLASS
# ==========================================
class PowerGridDataset(InMemoryDataset):
    def __init__(self, root, dataset_type, transform=None, pre_transform=None):
        self.dataset_type = dataset_type 
        super().__init__(root, transform, pre_transform)
        # Add weights_only=False right here:
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'IEEE300')

    @property
    def processed_file_names(self):
        return [f'{self.dataset_type}_data.pt']

    def process(self):
        data_list = []
        
        print(f"Loading {self.dataset_type} CSVs into memory...")
        numGraphs, numVariableNodes, numMeasurements, estimate_rows, jacobian_rows, measurement_rows, variance_rows, covariance_rows = read_from_wls_se_files(self.dataset_type, self.raw_dir)
        flat_start = torch.ones((numVariableNodes, 1), dtype=torch.float)
        
        encoded_indices = torch.tensor(encode_variable_node_indices(numVariableNodes), dtype=torch.float)

        print(f"Vectorizing {numGraphs} graphs into HeteroData objects...")
        for iGraph in range(numGraphs):
            if iGraph % 100 == 0 and iGraph > 0:
                print(f"Processed {iGraph}/{numGraphs} graphs...")

            data = HeteroData()
            
            # Variables
            est_row = [float(x) for x in estimate_rows[iGraph]]
            x_vars = torch.tensor(est_row, dtype=torch.float).view(-1, 1)
            data['variable'].x = torch.cat([flat_start, encoded_indices], dim=-1)
            
            # Factors
            meas = torch.tensor([float(x) for x in measurement_rows[iGraph]], dtype=torch.float)
            var = torch.log10(torch.tensor([float(x) for x in variance_rows[iGraph]], dtype=torch.float)) / 10.0
            cov_temp = torch.tensor([float(x) for x in covariance_rows[iGraph]], dtype=torch.float)
            cov = torch.repeat_interleave(cov_temp, 2) * 10.0
            data['factor'].x = torch.stack([meas, var, cov], dim=-1)
            
            # Bipartite Edges
            start_idx = iGraph * numMeasurements
            end_idx = start_idx + numMeasurements
            jacob_block = torch.tensor(jacobian_rows[start_idx:end_idx], dtype=torch.float)
            edge_index_fv = torch.nonzero(torch.abs(jacob_block) > 0.0001).t().contiguous()
            data['factor', 'to', 'variable'].edge_index = edge_index_fv
            data['variable', 'to', 'factor'].edge_index = edge_index_fv.flip(0)
            
            # Augmented Edges
            binary_jacob = (torch.abs(jacob_block) > 0.0001).float()
            adj_vv = torch.matmul(binary_jacob.t(), binary_jacob)
            edge_index_vv = torch.nonzero(adj_vv > 0).t().contiguous()
            data['variable', 'to', 'variable'].edge_index = edge_index_vv

            # Ground Truth Target (Assuming estimate_rows contains the target states for training)
            # You will need to adjust this if your targets are in a different file
            data['variable'].y = x_vars

            data_list.append(data)
            
        print("Saving compiled dataset to disk...")
        torch.save(self.collate(data_list), self.processed_paths[0])
if __name__ == "__main__":
    print("Starting PyG Dataset generation...")
    
    # Instantiating these will trigger the process() method for each set
    # Make sure the 'root' matches where your 'data_from_wls_se_solver' folder is located
    train_dataset = PowerGridDataset(root='./', dataset_type='train')
    val_dataset = PowerGridDataset(root='./', dataset_type='validation')
    test_dataset = PowerGridDataset(root='./', dataset_type='test')
    
    print(f"Success! Train graphs: {len(train_dataset)}")
    print(f"Success! Validation graphs: {len(val_dataset)}")
    print(f"Success! Test graphs: {len(test_dataset)}")