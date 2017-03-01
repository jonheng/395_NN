% Task 3: Using InitialLR of 0.1 as found from previous task
filename = {'figures/clsfError3-#-#.png','figures/loss3-#-#.png'};
scalingFactor_list = [0.999, 0.99, 0.9, 0.8];
lrEpochThres_list = [5, 10, 50];

for idy = 1:length(lrEpochThres_list)
    lrEpochThres = lrEpochThres_list(idy);
    filename{1} = strcat('figures/clsfError3-1-', num2str(idy), '.png');
    filename{2} = strcat('figures/loss3-1-', num2str(idy), '.png');
    task3NNFunction(filename,1,'N/A',lrEpochThres);
    for idx = 1:length(scalingFactor_list)
        scalingFactor = scalingFactor_list(idx);
        filename{1} = strcat('figures/clsfError3-2-', num2str(idy), '-', num2str(idx), '.png');
        filename{2} = strcat('figures/loss3-2-', num2str(idy), '-', num2str(idx), '.png');
        task3NNFunction(filename,2,scalingFactor,lrEpochThres);
    end
    filename{1} = strcat('figures/clsfError3-3-', num2str(idy), '.png');
    filename{2} = strcat('figures/loss3-3-', num2str(idy), '.png');
    task3NNFunction(filename,3,'N/A',lrEpochThres);
end
