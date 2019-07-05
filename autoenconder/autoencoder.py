#MLPs para autoassociativas 

#   Redes treinadas para reconhecer especificamente uma qualidade de entradas
#   Exemplo: rede especializada em reconhecer caes, deve ter como target
#imagens de caes.
#   No caso de mesma entrada para todas as redes, a rede especializada
#na classe que corresponde aquela entra apresentara o menor erro quadrático médio
#logo, podemos associar esta classe a essa rede 


#Para criar uma rede
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure import BiasUnit


#Para Classificacao
from pybrain.datasets            import ClassificationDataSet
from pybrain.tools.validation    import ModuleValidator
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

#Para importar tabela do excel
import pandas as pd

#Cofigurando Dataset
database = pd.read_csv('../databases/iris_database.csv')

dataframeA = ClassificationDataSet(4, 3, nb_classes=4, class_labels=['C1', 'C2', 'C3','C4'])
dataframeB = ClassificationDataSet(4, 3, nb_classes=4, class_labels=['C1', 'C2', 'C3','C4'])
dataframeC = ClassificationDataSet(4, 3, nb_classes=4, class_labels=['C1', 'C2', 'C3','C4'])

#As entradas e saidas das redes sao as mesmas, mas cada rede treina com
#os dados de apenas uma classe

#a base de dados A corresponde somente ao dados da classe 1
dataframeA.setField('input', database.get(['In1','In2','In3','In4'])[0:50]);
dataframeA.setField('target', database.get(['In1','In2','In3','In4'])[0:50]);
#a base de dados B corresponde somente ao dados da classe 2
dataframeB.setField('input', database.get(['In1','In2','In3','In4'])[50:100]);
dataframeB.setField('target', database.get(['In1','In2','In3','In4'])[50:100]);
#a base de dados C corresponde somente ao dados da classe 3
dataframeC.setField('input', database.get(['In1','In2','In3','In4'])[100:150]);
dataframeC.setField('target', database.get(['In1','In2','In3','In4'])[100:150]);



#separa os dados de cada base de dados, entre treino e teste a uma razao de 75/25 
tstdataA, trndataA = dataframeA.splitWithProportion(0.25)
tstdataB, trndataB = dataframeB.splitWithProportion(0.25)
tstdataC, trndataC = dataframeC.splitWithProportion(0.25)

#Criando 3 novas redes, com:
# as camadas de entrada com 4 neuronios
# as camadas escondidda com 2 neuronios
# as camadas de saida com 4 neuronios
neural_netA = buildNetwork( 4, 2, 4, outclass=LinearLayer )
neural_netB = buildNetwork( 4, 2, 4, outclass=LinearLayer )
neural_netC = buildNetwork( 4, 2, 4, outclass=LinearLayer )

#pesos antes do treinamento
# print (neural_netA.params)
# print (neural_netB.params)
# print (neural_netC.params)
#Configurando o treinamento da rede
trainerA = BackpropTrainer(neural_netA, trndataA, momentum=0.1, verbose=True, weightdecay=0.01)
trainerB = BackpropTrainer(neural_netB, trndataB, momentum=0.1, verbose=True, weightdecay=0.01)
trainerC = BackpropTrainer(neural_netC, trndataC, momentum=0.1, verbose=True, weightdecay=0.01)

#Treinar a rede, passando a base de dados 20 vezes ela rede
trainerA.trainEpochs(100)
trainerB.trainEpochs(100)
trainerC.trainEpochs(100)

#testando o erro quadratico medio para o datasetA\classe 1
val = ModuleValidator()
print('EQM para dados da classe A')
print('Rede A:')
print(val.MSE(neural_netA, tstdataA))
print('Rede B:')
print(val.MSE(neural_netB, tstdataA))
print('Rede C:')
print(val.MSE(neural_netC, tstdataA))
#testando o erro quadratico medio para o datasetA\classe 1
print('EQM para dados da classe B')
print('Rede A:')
print(val.MSE(neural_netA, tstdataB))
print('Rede B:')
print(val.MSE(neural_netB, tstdataB))
print('Rede C:')
print(val.MSE(neural_netC, tstdataB))
#testando o erro quadratico medio para o datasetA\classe 1
print('EQM para dados da classe C')
print('Rede A:')
print(val.MSE(neural_netA, tstdataC))
print('Rede B:')
print(val.MSE(neural_netB, tstdataC))
print('Rede C:')
print(val.MSE(neural_netC, tstdataC))
