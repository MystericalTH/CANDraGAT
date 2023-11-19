import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as PyGGMaxPool
from candragat.attentivefp import *

def get_drug_model(modelname:str,config:dict):  
    
    if modelname == 'GCN':
        return  GCNNet(
                num_features_xd=75,
                output_dim=config['drug_output_size']
            )
    elif modelname == 'GAT':
        return  GATNet(
                num_features_xd=75,
                output_dim=config['drug_output_size']
            )
    elif modelname == 'FragAttentiveFP':
        return  MolPredFragFPv8(
                atom_feature_size=72,  # 'atom_feature_size': 39
                bond_feature_size=10,  # 'bond_feature_size': 10
                FP_size=150,         # 'FP_size': 150
                atom_layers=3,     # 'atom_layers':3
                mol_layers=2,      # 'mol_layers':2
                DNN_layers=[256],   # 'DNNLayers':[512]
                output_size=config['drug_output_size'],
                drop_rate=config['drop_rate'],    # 'drop_rate':0.2
            )          # drugmodel = molpredfragfpv8
    elif modelname == 'AttentiveFP':
        return  MolPredAttentiveFP(
            atom_feature_size=72,  # 'atom_feature_size': 39
            bond_feature_size=10,  # 'bond_feature_size': 10
            FP_size=150,         # 'FP_size': 150
            atom_layers=3,     # 'atom_layers':3
            mol_layers=2,      # 'mol_layers':2
            DNN_layers=[256],   # 'DNNLayers':[512]
            output_size=config['drug_output_size'],
            drop_rate=config['drop_rate'],    # 'drop_rate':0.2
        )    
    elif modelname == 'AGMI':
        raise NotImplementedError("wait for implementation")
    else:
        raise NotImplementedError

# Drug Modules
class DNN(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.LayerList = nn.ModuleList()

        self.layer_sizes = layer_sizes
        self.Drop = nn.Dropout(p=0.2)

        if len(layer_sizes) == 0:
            self.FC = nn.Linear(input_size, output_size)
        else:
            for i in range(len(layer_sizes)):
                if i == 0:
                    self.LayerList.append(
                        nn.Linear(input_size, layer_sizes[i]))
                    self.LayerList.append(nn.Tanh())
                else:
                    self.LayerList.append(
                        nn.Linear(layer_sizes[i-1], layer_sizes[i]))
                    self.LayerList.append(nn.ReLU())
            self.Output = nn.Linear(layer_sizes[-1], output_size)

    def forward(self, x):
        if len(self.layer_sizes) == 0:
            x = self.FC(x)
        else:
            for num, layer in enumerate(self.LayerList):
                x = layer(x)
            x = self.Drop(x)
            x = self.Output(x)
        return x

class MolPredAttentiveFP(nn.Module):
    def __init__(
            self,
            atom_feature_size,
            bond_feature_size,
            FP_size,
            atom_layers,
            mol_layers,
            DNN_layers,
            output_size,
            drop_rate,
    ):
        super(MolPredAttentiveFP, self).__init__()
        self.AtomEmbedding = AttentiveFP_atom(
            atom_feature_size = atom_feature_size,
            bond_feature_size = bond_feature_size,
            FP_size = FP_size,
            layers = atom_layers,
            droprate = drop_rate
        )
        self.MolEmbedding = AttentiveFP_mol(
            layers = mol_layers,
            FP_size = FP_size,
            droprate = drop_rate
        )
        self.Classifier = DNN(
            input_size = FP_size,
            output_size = output_size,
            layer_sizes = DNN_layers,
        )

    def forward(self, Input):
        [atom_features, bond_features, atom_neighbor_list, bond_neighbor_list, atom_mask] = Input

        atom_FP = self.AtomEmbedding(atom_features, bond_features, atom_neighbor_list, bond_neighbor_list)
        mol_FP, _ = self.MolEmbedding(atom_FP, atom_mask)
        prediction = self.Classifier(mol_FP)
        return prediction

class MolPredFragFPv8(nn.Module):
    def __init__(self,
                 atom_feature_size,
                 bond_feature_size,
                 FP_size,
                 atom_layers,
                 mol_layers,
                 DNN_layers,
                 output_size,
                 drop_rate,
                 ):
        super().__init__()
        self.AtomEmbedding = AttentiveFP_atom(
            atom_feature_size=atom_feature_size,
            bond_feature_size=bond_feature_size,
            FP_size=FP_size,
            layers=atom_layers,
            droprate=drop_rate
        )   # For Frags and original mol_graph
        self.MolEmbedding = AttentiveFP_mol(
            layers=mol_layers,
            FP_size=FP_size,
            droprate=drop_rate
        )  # MolEmbedding module can be enabled repeatedly
        self.Classifier = DNN(
            input_size=4*FP_size,
            output_size=output_size,
            layer_sizes=DNN_layers,
        )
        self.AtomEmbeddingHigher = AttentiveFP_atom(
            atom_feature_size=FP_size,
            bond_feature_size=bond_feature_size,
            FP_size=FP_size,
            layers=atom_layers,
            droprate=drop_rate
        )  # For Junction Tree
        # self.InformationFenabler =
        self._name = 'FragAttentiveFP'
    @property
    def _device(self):
        return next(self.parameters()).device

    def forward(self, Input):
        [atom_features,
         bond_features,
         atom_neighbor_list_origin,
         bond_neighbor_list_origin,
         atom_mask_origin,
         atom_neighbor_list_changed,
         bond_neighbor_list_changed,
         frag_mask1,
         frag_mask2,
         bond_index,
         JT_bond_features,
         JT_atom_neighbor_list,
         JT_bond_neighbor_list,
         JT_mask] = Input

        # layer origin
        atom_FP_origin = self.AtomEmbedding(atom_features,
                                            bond_features, 
                                            atom_neighbor_list_origin, 
                                            bond_neighbor_list_origin)
        mol_FP_origin, _ = self.MolEmbedding(atom_FP_origin, atom_mask_origin)

        # layer Frag:
        atom_FP = self.AtomEmbedding(
            atom_features, bond_features, atom_neighbor_list_changed, bond_neighbor_list_changed)
        mol_FP1, activated_mol_FP1 = self.MolEmbedding(atom_FP, frag_mask1)
        mol_FP2, activated_mol_FP2 = self.MolEmbedding(atom_FP, frag_mask2)
        # mol_FP1, mol_FP2 are enabled to input the DNN module.
        # activated_mol_FP1 and activated_mol_FP2 are enabled to calculate the mol_FP
        # size: [batch_size, FP_size]
        ##################################################################################
        # Junction Tree Construction
        # construct a higher level graph: Junction Tree

        # Construct atom features of JT:
        batch_size, FP_size = activated_mol_FP1.size()
        pad_node_feature = torch.zeros(batch_size, FP_size).to(self._device)
        JT_atom_features = torch.stack(
            [activated_mol_FP1, activated_mol_FP2, pad_node_feature], dim=1)

        # Junction Tree Construction complete.
        ##################################################################################
        # layer Junction Tree: calculate information of the junction tree of Frags

        atom_FP_super = self.AtomEmbeddingHigher(JT_atom_features,
                                                 JT_bond_features,
                                                 JT_atom_neighbor_list,
                                                 JT_bond_neighbor_list)
        JT_FP, _ = self.MolEmbedding(atom_FP_super, JT_mask)
        entire_FP = torch.cat([mol_FP1, mol_FP2, JT_FP, mol_FP_origin], dim=-1)

        prediction = self.Classifier(entire_FP)
        return prediction

class GCNNet(nn.Module):
    
    def __init__(self, n_output=100, n_filters=32, embed_dim=128, num_features_xd=75, num_features_xt=25, output_dim=128, dropout=0.5):

        super().__init__()
        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # cell line feature
        self.conv_xt_1 = nn.Conv1d(
            in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(
            in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(
            in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(3)
        self.fc1_xt = nn.Linear(2944, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, self.n_output)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = PyGGMaxPool(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)
        return x

class GATNet(nn.Module):
    def __init__(self, num_features_xd=75, heads=10, n_output=1,
                 n_filters=32, output_dim=128, dropout=0.2):
        super().__init__()
        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd,
                            heads=heads, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # cell line feature
        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(3)
        self.fc1_xt = nn.Linear(2944, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = PyGGMaxPool(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)
        return x

# Omics Modules

class MutNet(nn.Module):
    def __init__(self,input_size, output_size, drop_rate):
        super().__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv1d(1, 50, 700, stride=5)
        self.maxpool1 = nn.MaxPool1d(10)
        self.conv2 = nn.Conv1d(50, 30, 5, stride=2)
        self.maxpool2 = nn.MaxPool1d(5)
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(self.conv_flatten_size, output_size) # DeepCDR = 2010, 2700
        self.fc = nn.Linear(self.conv_flatten_size, output_size) # DeepCDR = 2010, 2700
        self.drop_rate = drop_rate

    def forward(self, Input):
        x = self.maxpool1(F.relu(self.conv1(Input)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.dropout(self.fc(x), self.drop_rate)
        return x
    @property
    def conv_flatten_size(self):
        # raise NotImplementedError("Fix here") # NOTE
        layer_after_conv1 = (self.input_size-700)//5 + 1
        layer_after_maxpool1 = layer_after_conv1//10
        layer_after_conv2 = (layer_after_maxpool1-5)//2 + 1
        layer_after_maxpool2 = layer_after_conv2//5
        final_size = 30*layer_after_maxpool2
        return final_size

class ExprNet(nn.Module):
    def __init__(self,input_size, output_size, drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, output_size)
        self.drop_rate = drop_rate

    def forward(self,Input):
        x = self.bn(self.fc1(Input))
        x = F.dropout(F.relu(x), self.drop_rate)
        x = F.dropout(F.relu(self.fc2(x)), self.drop_rate)
        return x

class CNVNet(nn.Module):
    def __init__(self,input_size, output_size, drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, output_size)
        self.drop_rate = drop_rate

    def forward(self,Input):
        x = F.relu(self.fc1(Input))
        x = F.dropout(self.bn(x), self.drop_rate)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.drop_rate)
        return x

class MethNet(nn.Module):
    def __init__(self,input_size, output_size, drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256) #input_size = omicsdata[2].shape[1]
        self.bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, output_size)
        self.drop_rate = drop_rate
    def forward(self,Input):
        x = F.relu(self.fc1(Input))
        x = F.dropout(self.bn(x), self.drop_rate)
        x = F.dropout(F.relu(self.fc2(x)), self.drop_rate)
        return x

class MultiOmicsMolNet(nn.Module):

    def __init__(self,
                 drug_nn,
                 drop_rate,
                 drug_output_size,
                 omics_output_size,
                 input_size_list,
                 args,
                 ):

        super().__init__()
        self.omics_output_size = omics_output_size
        self.drug_output_size = drug_output_size
        self.mut_nn = MutNet(input_size_list[0], self.omics_output_size, drop_rate)
        self.expr_nn = ExprNet(input_size_list[1], self.omics_output_size, drop_rate)
        self.meth_nn = MethNet(input_size_list[2], self.omics_output_size, drop_rate)
        self.cnv_nn = CNVNet(input_size_list[3], self.omics_output_size, drop_rate)
        self.drug_nn = drug_nn
        self.args = args

        self.omics_fc = nn.Sequential(
            nn.Linear(self.omics_output_size*(self.args['enable_mut'] + self.args['enable_expr'] + self.args['enable_meth'] + self.args['enable_cnv']), 
                        self.omics_output_size),
            nn.LeakyReLU()
        )

        self.out = nn.Sequential(
            nn.Linear(self.omics_output_size+self.drug_output_size, 100),
            nn.LeakyReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(50, 1),
        )

    def forward(self, Input):
        [Mut, Expr, Meth, CNV], Drug = Input
        
        MultiOmics_layer = []

        if self.args['enable_mut']:
            Mut_layer = self.mut_nn(Mut)
            MultiOmics_layer.append(Mut_layer)
        if self.args['enable_expr']:
            Expr_layer = self.expr_nn(Expr)
            MultiOmics_layer.append(Expr_layer)
        if self.args['enable_meth']:
            Meth_layer = self.meth_nn(Meth)
            MultiOmics_layer.append(Meth_layer)
        if self.args['enable_cnv']:
            CNV_layer = self.cnv_nn(CNV)
            MultiOmics_layer.append(CNV_layer)
            
        MultiOmics_layer = torch.cat(MultiOmics_layer, dim=1)
        preout_layer = self.omics_fc(MultiOmics_layer)

        if self.args['enable_drug']:
            Drug_layer = self.drug_nn(Drug)
            preout_layer = torch.cat([preout_layer,Drug_layer], dim=1)

        prediction = self.out(preout_layer)

        if self.args['task'] == 'clas':
            prediction = torch.sigmoid(prediction)

        return prediction
    
    def forward_validation(self, Input):
        [Mut, Expr, Meth, CNV], DrugTensor = Input
        MultiOmics_layer = []
        
        if self.args['enable_mut']:
            Mut_layer = self.mut_nn(Mut)
            MultiOmics_layer.append(Mut_layer)
        if self.args['enable_expr']:
            Expr_layer = self.expr_nn(Expr)
            MultiOmics_layer.append(Expr_layer)
        if self.args['enable_meth']:
            Meth_layer = self.meth_nn(Meth)
            MultiOmics_layer.append(Meth_layer)
        if self.args['enable_cnv']:
            CNV_layer = self.cnv_nn(CNV)
            MultiOmics_layer.append(CNV_layer)
            
        MultiOmics_layer = torch.cat(MultiOmics_layer, dim=1)
        preout_layer = self.omics_fc(MultiOmics_layer)

        if self.args['enable_drug']:
            preout_layer = torch.cat([preout_layer,DrugTensor], dim=1)

        prediction = self.out(preout_layer)
        return prediction


