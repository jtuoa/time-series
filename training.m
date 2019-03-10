%%%%%%%%%% Set up Nested 5-fold Cross Validation %%%%%%%%%%%%
%%Script 'main.m' should be called to set up the workspace before running
%%this script

infection = [patientDB.infection];
idx_inf = find(infection==1);
idx_noinf = find(infection==0);

%K=10; %Number of CV folds (nested CV)
%Create a set of new (undersampled) datasets as follows: 
%Keep all patient rows with infection=1 (idx_inf). From patients with 
%infection=2 (majority class), we subsample (without replacement) the same 
%number of samples as for infection=1. Do this as many times as possible to build maximum possible
%number of datasets. If there are any remaining cases from majority class,
%divide them equally (and randomly) between the D balanced datasets we
%built.

minClassRatio = 0.7; %i.e. at least 70% of data should be from minority class
[datasets,numD] = CreateBalancedDatasets(idx_inf,idx_noinf,minClassRatio);


PARAMS_LENGTH = 0;
K = 5; %number of cross validation folds for nested CV
acc = zeros(1,K*numD); %5*3=15 accuracy scores (Report average of these)

for d=1:numD
    D = patientDB(datasets(d,:)); %D is a subset of patientDB, that I want to consider in this loop
    
    %Partition D into K folds, keep 1 fold for testing, K-1 for training
    nRows = size(D,2);
    cv = cvpartition(nRows,'KFold',K,'Stratify',false);
    
    for kOut=1:cv.NumTestSets %Outer CV loop
        trainIdx = cv.training(kOut);
        testIdx = cv.test(kOut);
        
        %Train model using D(trainIdx)
        %%%%%%Within training (inner CV loop), divide again into Train-Validate folds for
        %%%%%%model selection (keep track of best params)
        
        trainSet = D(trainIdx);
        testSet = D(testIdx);
        nRowsInner = size(trainSet,2);
        cvInner = cvpartition(nRowsInner,'KFold',K,'Stratify',false); 
        
        for kIn=1:cvInner.NumTestSets
            inTrainIdx = cvInner.training(kIn);
            inTestIdx = cvInner.test(kIn);
            
            innerTrainSet = trainSet(inTrainIdx);
            innerTestSet = trainSet(inTestIdx);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%ONLY FOR DEBUGGING....DELETE LATER!!!
            %innerTrainSet = innerTestSet;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            SVMModel=[];
            
            %Step1: For all samples in innerTrainSet, z-normalize TestVal
            %arrays (separately for each test, and store the mean and stdev)
            for i=1:length(innerTrainSet)
                numTests=length(innerTrainSet(i).testdata);
                %Normalize the test values:
                for j=1:numTests
                    meanTestVal = mean([innerTrainSet(i).testdata(j).TestVal]);
                    stdTestVal = std([innerTrainSet(i).testdata(j).TestVal]);
                    [innerTrainSet(i).testdata(j).TestVal]=([innerTrainSet(i).testdata(j).TestVal]-meanTestVal)./stdTestVal;
                    innerTrainSet(i).testdata(j).meanTestVal = meanTestVal;
                    innerTrainSet(i).testdata(j).stdTestVal = stdTestVal;
                    
                    %For each test, fit a GP and compute the inferred GP
                    %hyperparameters. Store these in struct to represent
                    %the current time series for use in classification:
                    meanfunc = [];                    % empty: don't use a mean function
                    covfunc = @covSEiso;              % Squared Exponental covariance function
                    likfunc = @likGauss;              % Gaussian likelihood
                    hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);
                    x = [innerTrainSet(i).testdata(j).MinsToSurgery];
                    y = [innerTrainSet(i).testdata(j).TestVal];
                    hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, -x, y);
%                     [mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, -x, y, -x);
%                     f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
%                     fill([-x; flipdim(-x,1)], f, [7 7 7]/8);
%                     hold on; plot(-x, mu); plot(-x, y, '+');
                    innerTrainSet(i).testdata(j).gphyperparams = struct2array(hyp2);
                    %innerTrainSet(i).testdata(j).hyplength = length(struct2array(hyp2));
                    PARAMS_LENGTH = 2+length(struct2array(hyp2)); %Test mean, Test Std, Test GP hyperparams
                end 
            end
            
            %Concatenate all relevant data into a row of features for each
            %patient...Build an SVM-RBF model to classify each patient given
            %the training label 'infection'
            for i=1:length(innerTrainSet)
                numTests=length(innerTrainSet(i).testdata);
                innerTrainSet(i).paramvec = zeros(1,TOTAL_NUMBER_OF_TESTS*PARAMS_LENGTH);
                for j=1:numTests
                    tid=innerTrainSet(i).testdata(j).TestType;
                    startIndex=(tid-1)*PARAMS_LENGTH+1;
                    endIndex=startIndex+PARAMS_LENGTH-1;
                    params = [innerTrainSet(i).testdata(j).meanTestVal,innerTrainSet(i).testdata(j).stdTestVal,innerTrainSet(i).testdata(j).gphyperparams];
                    innerTrainSet(i).paramvec(startIndex:endIndex) = params;
                end
                innerTrainSet(i).trainVec = [innerTrainSet(i).sex, innerTrainSet(i).yob, innerTrainSet(i).numTimeSeries, innerTrainSet(i).testsperformed, innerTrainSet(i).paramvec];
            end
            
            
            %%%%%%%%%%%%%%%%%Doing same data preparation as for
            %%%%%%%%%%%%%%%%%innerTrainSet, but for innerTestSet
            for i=1:length(innerTestSet)
                numTests=length(innerTestSet(i).testdata);
                %Normalize the test values:
                for j=1:numTests
                    meanTestVal = mean([innerTestSet(i).testdata(j).TestVal]);
                    stdTestVal = std([innerTestSet(i).testdata(j).TestVal]);
                    [innerTestSet(i).testdata(j).TestVal]=([innerTestSet(i).testdata(j).TestVal]-meanTestVal)./stdTestVal;
                    innerTestSet(i).testdata(j).meanTestVal = meanTestVal;
                    innerTestSet(i).testdata(j).stdTestVal = stdTestVal;
                    
                    %For each test, fit a GP and compute the inferred GP
                    %hyperparameters. Store these in struct to represent
                    %the current time series for use in classification:
                    meanfunc = [];                    % empty: don't use a mean function
                    covfunc = @covSEiso;              % Squared Exponental covariance function
                    likfunc = @likGauss;              % Gaussian likelihood
                    hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);
                    x = [innerTestSet(i).testdata(j).MinsToSurgery];
                    y = [innerTestSet(i).testdata(j).TestVal];
                    hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, -x, y);
                    innerTestSet(i).testdata(j).gphyperparams = struct2array(hyp2);
                    %innerTrainSet(i).testdata(j).hyplength = length(struct2array(hyp2));
                    PARAMS_LENGTH = 2+length(struct2array(hyp2)); %Test mean, Test Std, Test GP hyperparams
                end 
            end
            
            %Concatenate all relevant data into a row of features for each
            %patient.
            for i=1:length(innerTestSet)
                numTests=length(innerTestSet(i).testdata);
                innerTestSet(i).paramvec = zeros(1,TOTAL_NUMBER_OF_TESTS*PARAMS_LENGTH);
                for j=1:numTests
                    tid=innerTestSet(i).testdata(j).TestType;
                    startIndex=(tid-1)*PARAMS_LENGTH+1;
                    endIndex=startIndex+PARAMS_LENGTH-1;
                    params = [innerTestSet(i).testdata(j).meanTestVal,innerTestSet(i).testdata(j).stdTestVal,innerTestSet(i).testdata(j).gphyperparams]
                    innerTestSet(i).paramvec(startIndex:endIndex) = params;
                end
                innerTestSet(i).trainVec = [innerTestSet(i).sex, innerTestSet(i).yob, innerTestSet(i).numTimeSeries, innerTestSet(i).testsperformed, innerTestSet(i).paramvec];
            end
            
            
            
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            X=[];
            y=[];
            for i=1:length(innerTrainSet)
                X = [X; innerTrainSet(i).trainVec];
                y = [y; innerTrainSet(i).infection];
            end
            
            Xtest=[];
            ytest=[];
            for i=1:length(innerTestSet)
                Xtest = [Xtest; innerTestSet(i).trainVec];
                ytest = [ytest; innerTestSet(i).infection];
            end
            
            [Z,mu,sigma] = zscore(X);
            imagesc(Z);
            figure;
            [pccoeff,pcvec] = pca(Z);
            %plot(pccoeff);
            Xpca5 = Z*pcvec(:,1:5);
            scatter(Xpca5(:,1),Xpca5(:,2));
            
            SVMModel = fitcsvm(X,y,'KernelFunction','rbf','Standardize',true);
            %clear innerTrainSet;
            
            [label,score] = predict(SVMModel,Xtest); %this should be InnerTestSet, but passing X just for debugging
            errors = ytest - label; %+1 is false negative, -1 is false positive
            falsePositive = length(find(errors==-1)); %This is a patient without infection, classified as with infection
            falseNegative = length(find(errors==1)); %This is a patient with infection, classified as without infection
            accuracy = length(find(errors==0))/length(errors);
        end %End of Inner CV loop
        

        
        
        %Test model using D(testIdx)
        %store accuracy in acc(k*d)
            
    end %End of Outer CV loop 
    
end %End of balanced data sets loop 

