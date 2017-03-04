% Task 7
filename = {'figures/clsfError7-#.png','figures/loss7-#.png'};
activationFunction = {'leakyReLu','ReLu'};
valClsfError_cell={};
stats_cell={};

for i=1:2
    filename{1} = strcat('figures/clsfError7-',activationFunction{i},'.png');
    filename{2} = strcat('figures/loss7-',activationFunction{i},'.png');
    [clsfError,stats] = task7NNFunction(filename,activationFunction{i});
    valClsfError_cell{i} = clsfError;
    stats_cell{i} = stats;
end

f=figure();
plot(1:length(valClsfError_cell{1}),valClsfError_cell{1},1:length(valClsfError_cell{2}),valClsfError_cell{2});
legend('leakyReLu','ReLu');
title('Classification errors on validation set for different activation functions')
xlabel('Number of Epochs')
ylabel('Classification Error')
saveas(f,'clsfError7-activationFunction.png');
close all

