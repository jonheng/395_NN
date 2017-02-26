% Task 2: Optimising initial learning rates
filename = {'figures/clsfError2-1.png','figures/loss2-1.png'};
initialLR_list = [0.5,0.1,0.01,0.001];

for idx = 1:length(initialLR_list)
    initialLR = initialLR_list(idx);
    filename{1}(20)=num2str(idx);
    filename{2}(15)=num2str(idx);
    task2NNFunction(filename,initialLR);
end
