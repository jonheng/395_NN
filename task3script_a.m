% Task 3: Using InitialLR of 0.1 as found from previous task
filename = {'figures/clsfError3-#-#.png','figures/loss3-#-#.png'};
scalingFactor_list = [0.9];
lrEpochThres_list = [25, 10, 5];

C1 = {};
for idx = 1:length(lrEpochThres_list)
    lrEpochThres = lrEpochThres_list(idx);
    filename{1} = strcat('figures/clsfError3-1-', num2str(idx), '.png');
    filename{2} = strcat('figures/loss3-1-', num2str(idx), '.png');
    C1{idx} = task3NNFunction(filename,1,'N/A',lrEpochThres);
end

C2 = {};
scalingFactor = scalingFactor_list(1);

for idx = 1:length(lrEpochThres_list)
    lrEpochThres = lrEpochThres_list(idx);
    filename{1} = strcat('figures/clsfError3-2-', num2str(idx), '.png');
    filename{2} = strcat('figures/loss3-2-', num2str(idx), '.png');
    C2{idx} = task3NNFunction(filename,2,scalingFactor,lrEpochThres);
end

C3 = {};
for idx = 1:length(lrEpochThres_list)
    lrEpochThres = lrEpochThres_list(idx);
    filename{1} = strcat('figures/clsfError3-3-', num2str(idx), '.png');
    filename{2} = strcat('figures/loss3-3-', num2str(idx), '.png');
    C3{idx} = task3NNFunction(filename,3,'N/A',lrEpochThres);
end
