function [Ds,numDs] = CreateBalancedDatasets(class1idx,class2idx,minClassRatio)
%CREATEBALANCEDDATASETS Function to create balanced datasets by
%undersampling from class2idx (Assumption: class1idx remain the same,
%class2idx gets undersampled to create all new datasets)
% Ds - array of vectors containing sample IDs for each new dataset
% numDs - total number of newly created datasets

numDs = floor(length(class2idx)/length(class1idx));
tempDs = [];

for d=1:numDs
    numSamples=length(class1idx);
    newC2idx = randperm(length(class2idx),numSamples);
    newD = [class1idx, class2idx(newC2idx)];
    newD = newD(randperm(length(newD)));
    tempDs(d,:) =  newD;
    class2idx = setdiff(class2idx, class2idx(newC2idx)); %remove newC2idx from class2idx
end

%Put remaining class2idx into datasets we just created (divided almost equally)
numExtraPerD = floor(length(class2idx)/numDs); %roughly the number of extra samples we need to assign to each of the new Ds
numMaxExtraAllowed = floor(length(class1idx)/minClassRatio) - length(class1idx);
if numExtraPerD > numMaxExtraAllowed
    numExtraPerD = numMaxExtraAllowed;
end

finalDs = [];

for d=1:numDs
    extraC2idx = randperm(length(class2idx),numExtraPerD);
    augmentedD = [tempDs(d,:), class2idx(extraC2idx)];
    augmentedD = augmentedD(randperm(length(augmentedD)));
    finalDs(d,:) =  augmentedD;
    class2idx = setdiff(class2idx, class2idx(extraC2idx)); %remove newC2idx from class2idx
end

Ds = finalDs;

end

