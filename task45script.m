% Tasks 4 & 5: Using InitialLR = 0.1, (schType = 2, scalingFactor = 0.99,
% lrEpochThres = 5), introduce dropoutType = 1 or
% weightPenaltyL1 = [0.1, 0.01, 0.001] or weightPenaltyL2 = [0.1, 0.01, 0.001]
filename = {'figures/clsfError#-#-#.png','figures/loss#-#-#.png'};

% idx = 1 (default) or idx = 2 (optimal)
schType_list = [1, 2];
scalingFactor_list = [1, 0.99];
lrEpochThres_list = [50, 5];

weightPenaltyL1_list = [0.01, 0.005, 0.001, 0.0005, 0.0001];
weightPenaltyL2_list = [0.01, 0.005, 0.001, 0.0005, 0.0001];

for idx = 1:length(lrEpochThres_list)
    weightPenaltyL1 = 0;
    weightPenaltyL2 = 0;
    schType = schType_list(idx);
    scalingFactor = scalingFactor_list(idx);
    lrEpochThres = lrEpochThres_list(idx);
    % obtain figures for no dropout w/ and w/o optimal update schedule
    filename{1} = strcat('figures/clsfError4-1-', num2str(idx), '.png');
    filename{2} = strcat('figures/loss4-1-', num2str(idx), '.png');
    task45NNFunction(filename,schType,scalingFactor,lrEpochThres,0,0,0);
    % obtain figures for dropout w/ and w/o optimal update schedule
    filename{1} = strcat('figures/clsfError4-2-', num2str(idx), '.png');
    filename{2} = strcat('figures/loss4-2-', num2str(idx), '.png');
    % when using dropout, we also introduce a higher InitialLR = 10, 
    % higher finalMomentum = 0.99, and maxNormConstraint = 3 as per
    % useSomeDefaultNNparams.m - SEE LINE 125 OF task45NNFunction.M
    task45NNFunction(filename,schType,scalingFactor,lrEpochThres,1,0,0);
    
    % obtain figures for L1 and L2 regularisation w/ and w/o optimal update
    % schedule
    for idy = 1:length(weightPenaltyL1_list)
        weightPenaltyL1 = weightPenaltyL1_list(idy);
        filename{1} = strcat('figures/clsfError5-1-', num2str(idy), '-', num2str(idx), '.png');
        filename{2} = strcat('figures/loss5-1-', num2str(idy), '-', num2str(idx), '.png');
        task45NNFunction(filename,schType,scalingFactor,lrEpochThres,0,weightPenaltyL1,0);
    end
    for idz = 1:length(weightPenaltyL2_list)
        weightPenaltyL2 = weightPenaltyL2_list(idz);
        filename{1} = strcat('figures/clsfError5-2-', num2str(idz), '-', num2str(idx), '.png');
        filename{2} = strcat('figures/loss5-2-', num2str(idz), '-', num2str(idx), '.png');
        task45NNFunction(filename,schType,scalingFactor,lrEpochThres,0,0,weightPenaltyL2);
    end
end