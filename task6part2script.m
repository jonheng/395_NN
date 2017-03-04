% Task 6
filename = {'figures/clsfError6-#.png','figures/loss6-#.png'};
hiddenLayerSize = [100 300 500 700 900];
clsfValError_cell={};
stats_cell={};

for i=1:length(hiddenLayerSize)
    size=hiddenLayerSize(i);
    filename{1} = strcat('figures/clsfError6-hls-',num2str(size),'.png');
    filename{2} = strcat('figures/loss6-hls-',num2str(size),'.png');
    [clsfError,stats] = task6NNpart2Function(filename,size);
    clsfValError_cell{i} = clsfError;
    stats_cell{i} = stats;
end

f=figure();
plot(1:length(clsfValError_cell{1}),clsfValError_cell{1},1:length(clsfValError_cell{2}),clsfValError_cell{2},1:length(clsfValError_cell{3}),clsfValError_cell{3},1:length(clsfValError_cell{4}),clsfValError_cell{4},1:length(clsfValError_cell{5}),clsfValError_cell{5});
legend('100','300','500','700','900');
title('Classification errors on validation set for different hidden layer sizes, number of hidden layers = 2')
xlabel('Hidden Layer Size')
ylabel('Classification Error')
saveas(f,'clsfError6-hiddenLayerSize.png');
close all

minValClsfError = zeros(1,5);
for i=1:5
    minValClsfError(i)=min(clsfValError_cell{i});
end

f=figure();
plot(hiddenLayerSize,minValClsfError);
title('Minimum classification error found during training against hidden layer size');
xlabel('Number of Epochs')
ylabel('Classification Error')
saveas(f,'clsfError6-minClsfError_vs_hiddenLayerSize.png');
close all