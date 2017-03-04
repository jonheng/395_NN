% Task 6
filename = {'figures/clsfError6-#.png','figures/loss6-#.png'};
numHiddenLayers = [1 2 3 4];
hiddenLayerSize = [100 300 500 700 900];
cell={};

for i=numHiddenLayers
    hiddenLayersShape = zeros(1,i)+500;
    hiddenActivationFunctions = {};
    for j=1:i
        hiddenActivationFunctions{j} = 'ReLu';
    end
    hiddenActivationFunctions{i+1} = 'softmax';
    filename{1} = strcat('figures/clsfError6-',num2str(i),'.png');
    filename{2} = strcat('figures/loss6-',num2str(i),'.png');
    clsfError = task6NNFunction(filename,hiddenLayersShape,hiddenActivationFunctions);
    cell{i} = clsfError;
end

f=figure();
plot(1:length(cell{1}),cell{1},1:length(cell{2}),cell{2},1:length(cell{3}),cell{3},1:length(cell{4}),cell{4});
legend('1 hidden layer','2 hidden layers','3 hidden layers','4 hidden layers');
title('Classification errors on validation set, hidden layer size=500')
xlabel('Number of hidden layers')
ylabel('Classification Error')
saveas(f,'clsfError6-numHiddenLayers.png');
close all

minValClsfError = zeros(1,4);
for i=1:4
    minValClsfError(i)=min(cell{i});
end

f=figure();
plot(numHiddenLayers,minValClsfError);
title('Minimum classification error found during training against number of hidden layers');
xlabel('Number of Epochs')
ylabel('Classification Error')
saveas(f,'clsfError6-minClsfError_vs_numHiddenLayers.png');
close all
