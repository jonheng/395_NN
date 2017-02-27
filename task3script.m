% Task 3: Using InitialLR of 0.1 as found from previous task
filename = {'figures/clsfError3-2-1.png','figures/loss3-2-1.png'};
scalingFactor_list = [0.999, 0.99, 0.9, 0.5];
% 50 epoch thres

for idx = 1:length(scalingFactor_list)
    scalingFactor = scalingFactor_list(idx);
    filename{1}(22)=num2str(idx);
    filename{2}(17)=num2str(idx);
    task3NNFunction(filename,2,scalingFactor,0);
end
