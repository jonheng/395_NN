function [clsfError,stats] = task6NNpart2Function(filename,hiddenLayersSize)
    type = 2; % 2 for NN

    % Loading data
    load('data4students.mat')
    train_x = datasetInputs{1};
    test_x = datasetInputs{2};
    val_x = datasetInputs{3};
    train_y = datasetTargets{1};
    test_y = datasetTargets{2};
    val_y = datasetTargets{3};

    % image dimension
    im_rows = 30;
    im_cols = 30;

    %input dimension of a single example
    inputSize = size(train_x,2);

    if type == 1 % AE
        outputSize = inputSize; 
        % last layer is linear
        % series of decreasing layers
        hiddenActivationFunctions = {'sigm','sigm','sigm','linear'};
        hiddenLayers = [1000 500 250 50 250 500 1000 outputSize];

    elseif type == 2 % classifier
        % output size is based on target vector
        outputSize = size(train_y,2);
        hiddenActivationFunctions = {'ReLu','ReLu','softmax'};
        hiddenLayers = [hiddenLayersSize hiddenLayersSize outputSize];
    end

    % parameters used for visualization of first layer weights
    % this creates a struct visParams with corresponding fields
    % visParams.noExamplesPerSubplot = 50; 
    % visParams.noSubplots = floor(hiddenLayers(1)/visParams.noExamplesPerSubplot);
    % visParams.col = 30; % number of image cols
    % visParams.row = 30; % number of image rows

    inputActivationFunction = 'linear';

    % normalize data
    % each image is independently normalized to 0 mean and 1 std dev
    train_x = normaliseData(inputActivationFunction,train_x,[]);
    val_x = normaliseData(inputActivationFunction,val_x,[]);
    test_x = normaliseData(inputActivationFunction,test_x,[]);

    %initialise NN params
    nn = paramsNNinit(hiddenLayers,hiddenActivationFunctions);
    % number of epochs
    nn.epochs = 250;

    % learning rate parameters (lrParams)
    % initial learning rate
    nn.trParams.lrParams.initialLR = 0.1;
    % threshold of learning rate decay (after set epochs)
    nn.trParams.lrParams.lrEpochThres = 10;
    % set learning rate update policy
    nn.trParams.lrParams.schedulingType = 1;
    % set scaling Factor
    nn.trParams.lrParams.scalingFactor = 0.99;

    % momentum parameters (momParams)
    % linear increase (only 1 type supported currently)
    nn.trParams.momParams.schedulingType = 1;
    % set epoch threshold when momentum starts increasing
    % initial value is 0.5 (can be changed)
    nn.trParams.momParams.momentumEpochLowerThres = 10;
    % set epoch threshold when momentum reaches final value 
    % default final value is 0.9 (can be changed)
    nn.trParams.momParams.momentumEpochUpperThres = 100;


    % set weight constraints (a.k.a. regularization constant)
    % linear weight penalty
    nn.weightConstraints.weightPenaltyL1 = 0;
    % squared weight penalty
    nn.weightConstraints.weightPenaltyL2 = 0.001;
    % max norm constraint (constraint ||w|| <= c, where c is usually 3 or 4)
    nn.weightConstraints.maxNormConstraint = 0;

    % show diagnostics to monitor training
    % setting to 1 displays the mean/st. dev. of neuron activations &
    % the ratio norm(delta w)/norm(w) (should be 0.01 - 0.0001)
    nn.diagnostics = 0;
    % show diagnostic every 'x' epochs
    nn.showDiagnostics = 10;

    % show training and validation loss plot
    %nn.showPlot = 1;

    % 0 for no dropout, 1 for Bernoulli dropout 
    nn.dropoutParams.dropoutType = 1;
    %Visible/input layer dropout rate (default=0.8)
    %nn.dropoutParams.dropoutPresentProbVis = 0.8
    %Hidden layer dropout rate (default=0.5)
    %nn.dropoutParams.dropoutPresentProbHid = 0.5

    % early stopping
    nn.earlyStopping = 1;
    % maximum number of increases in validation error until training stops
    nn.max_fail = 10;

    nn.type = type;

    % set type of weight initialisation (default = 8, 10 choices available)
    nn.weightInitParams.type = 8;

    % set training method
    % 1: SGD, 2: SGD with momentum, 3: SGD with nesterov momentum, 4: Adagrad, 5: Adadelta,
    % 6: RMSprop, 7: Adam
    nn.trainingMethod = 2;

    % initialise weights
    [W, biases] = initWeights(inputSize, nn.weightInitParams, hiddenLayers, hiddenActivationFunctions);

    nn.W = W;
    nn.biases = biases;

    % if dropout is used then use max-norm constraint and a
    %high learning rate + momentum with scheduling
    % see the function below for suggested values
    % nn = useSomeDefaultNNparams(nn);

    if type == 1 % AE
        [nn, Lbatch, L_train, L_val]  = trainNN(nn, train_x, train_x, val_x, val_x);
    elseif type == 2 % classifier
        [nn, Lbatch, L_train, L_val, clsfError_train, clsfError_val]  = trainNN(nn, train_x, train_y, val_x, val_y);
     end

    nn = prepareNet4Testing(nn);

    % visualise weights of first layer
    %figure()
    %visualiseHiddenLayerWeights(nn.W{1},visParams.col,visParams.row,visParams.noSubplots);


    if type == 1 % AE
        [stats, output, e, L] = evaluateNNperformance( nn, test_x, test_x);
    elseif type == 2 % classifier
        [stats, output, e, L] = evaluateNNperformance( nn, test_x, test_y);
    end

    f = figure();
    x_axis = find(clsfError_train);
    plot(x_axis,clsfError_train(x_axis),x_axis,clsfError_val(x_axis));
    legend('Training error','Validation error');
    plotname = strcat('Number of hidden layers=',num2str(2),', Hidden layer size=',num2str(hiddenLayersSize));
    title(plotname);
    xlabel('Number of Epochs');
    ylabel('Classification Error');
    saveas(f,filename{1})
    plot(x_axis,L_train(x_axis),x_axis,L_val(x_axis));
    legend('Training loss','Validation loss');
    title(plotname);
    xlabel('Number of Epochs');
    ylabel('Loss');
    saveas(f,filename{2})
    close all
    
    clsfError = clsfError_val(x_axis);
end