#MLP para solucao do problema de classificacao
#Verificar o comportamento da rede com 1, 5, 10, 15, 30 e 60 neuronios na camada escondida

#Para criar uma rede
from pybrain.structure import LinearLayer, SigmoidLayer
#Para Classificacao
from pybrain.datasets            import ClassificationDataSet
from pybrain.tools.validation    import ModuleValidator
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

#Para importar tabela do calc e escrever arquivos csv
import pandas as pd

#Importando dados da Iris de um arquivo .csv
database = pd.read_csv('../databases/iris_database.csv')
#Gerando uma tabela de dados com 4 entradas 1 saida e 3 classes
dataframe = ClassificationDataSet(4, 60, nb_classes=3, class_labels=['C1', 'C2', 'C3','C4'])

#A base de dados corresponde somente aos dados da classe 1, 2 e 3
dataframe.setField('input', database.get(['In1','In2','In3','In4'])[0:150]);
dataframe.setField('target', database.get(['Out'])[0:150]-1);

#Divide a base de dados entre dados de treino e teste a uma razao de 75/25
tstdata, trndata = dataframe.splitWithProportion(0.25)

#Criando uma nova rede
#com 4 neuronios de entrada
#com 1, 5, 10, 15, 30 ou 60 neuronios na camada escondidda
#com 1 neuronio na camada de saida
neural_net = buildNetwork( 4, 60, 1, outclass=LinearLayer )

#Configurando o treinamento da rede
trainer = BackpropTrainer(neural_net, trndata, momentum=0.1, verbose=True, weightdecay=0.01)

#Treinar a rede, passando a base de dados 100 vezes ela rede
trainer.trainEpochs(100)

#Retona a saida da rede para um dataset de entrada
val = ModuleValidator()
print('Saidas do teste para 60 neorônios na camada escondida')
print(val.calculateModuleOutput(neural_net,tstdata).transpose())
#Calcula o erro erro medio quadratico entre o target e a saída da rede
print('EQM para 60 neorônios na camada escondida: %.10f' %val.MSE(neural_net, tstdata) )

#Salva os dados em um arquivo .csv
pd.DataFrame(tstdata.getField('target')).to_csv("../results/mlp/test-targets-export.csv")
pd.DataFrame(val.calculateModuleOutput(neural_net,tstdata)).to_csv("../results/mlp/net-test-outputs-export.csv")

