%Algorithm pipeline:
%Step 1: Partition data set into sets of lab tests (time series) for individual
%patients
%Step 2: Choose a train-test split over patients for 5-fold nested cross
%validation
%Step 3: Z-Normalize each lab test separately..?
%Step 4: Use MTGPs and learn MTGP hyperparams from the training set of patients to build a SVM classifier 
clear all;
T = readtable('SSItimeseries.csv');
tests = readtable('TestNames.csv');

pidList = unique(T.PID);
numPatients = length(pidList);
patientDB = [];
minSeriesLength = 5; %If number of blood test values are less than 5, drop this test
TOTAL_NUMBER_OF_TESTS = 49;

%Filling patientDB with values from table
for p=1:numPatients
    pid = pidList(p);
    pidRows = find(T.PID==pid);
    patientDB(p).PID = pid;
    patientDB(p).infection = table2array(T(pidRows(1),'Infection'));
    sex = table2cell(T(pidRows(1),'Sex'));
    eye2 = eye(2);
    if strcmp(sex,'M') %one hot encoding the patient gender
        patientDB(p).sex = eye2(1,:);
    else
        patientDB(p).sex = eye2(2,:);
    end
    patientDB(p).yob = table2array(T(pidRows(1),'YoB'));
    patientDB(p).data = T(pidRows,{'TestType','NumAnswer','surgery_test_date'});
    
    %split the data struct into an array of structs, one per unique TestID
    %So for each TestID, we get an array containing surgery_test_time (minutes relative to surgery time) and
    %NumAnswer (value of this blood test at the given time stamp)
    testList = unique(patientDB(p).data.TestType);
    testList = setdiff(testList,0); %remove test number 0 from list
    numTests = length(testList);
    patientDB(p).numTimeSeries = numTests;
    patientDB(p).testsperformed = zeros(1,TOTAL_NUMBER_OF_TESTS);
    for t=1:numTests
        tid = testList(t);
        %patientDB(p).testsperformed(tid)=1;%Setting appropriate bit for this test in testsperformed vector
        testRows = find(patientDB(p).data.TestType==tid);
        ntestVals = length(testRows);
        patientDB(p).testsperformed(tid)=ntestVals;
        patientDB(p).testdata(t) = struct('TestType',tid,'MinsToSurgery',[],'TestVal',[]);
        for i=testRows
            x = table2array(patientDB(p).data(i,'surgery_test_date'));
            y = table2array(patientDB(p).data(i,'NumAnswer'));
            patientDB(p).testdata(t).MinsToSurgery = x;
            patientDB(p).testdata(t).TestVal = y;
        end
        if ( length(patientDB(p).testdata(t).MinsToSurgery) < minSeriesLength )
            patientDB(p).testdata(t).MinsToSurgery = [];
            patientDB(p).testdata(t).TestVal = [];
            patientDB(p).numTimeSeries = patientDB(p).numTimeSeries - 1;
        end
        [patientDB(p).testdata(t).MinsToSurgery,idx] = sort(patientDB(p).testdata(t).MinsToSurgery,'descend');
        patientDB(p).testdata(t).TestVal = patientDB(p).testdata(t).TestVal(idx);
    end
    for t=1:length(patientDB(p).testsperformed)
        if patientDB(p).testsperformed(t) < minSeriesLength
            patientDB(p).testsperformed(t) = 0;
        end
    end
    
end

patientDB = rmfield(patientDB,'data');

%plotPatientTests( patientDB(8) );

%%%%%%%%%%%%%%%%%%% DATA REDUCTION %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%First, go through whole dataset and for each patient REMOVE those blood tests that
%%%%have less than say 5 values recorded (??). Track how many you've had to
%%%%remove...Can we add this data as auxiliary (non-time series) data, and
%%%%somehow learn from it too? Maybe not...But we can add NumTimeSeries
%%%%(ie. Number of tests performed on this patient that each have at least
%%%%5 measurements)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
countNonemptyPatients = 0;

for p=1:numPatients
    if patientDB(p).numTimeSeries > 0
        countNonemptyPatients = countNonemptyPatients + 1;
    end
end

FullPatientDB = patientDB; %Save the original

goodRows = find([patientDB.numTimeSeries] > 0);
patientDB = patientDB(goodRows); %reduced patientDB

% min5TS = find([patientDB.numTimeSeries] >= 5);
% PDB_5ts = patientDB(min5TS);
% 
% min10TS = find([patientDB.numTimeSeries] >= 10);
% PDB_10ts = patientDB(min10TS);
% 
% patientDB = PDB_5ts; %patientDB is severely reduced now!

numPatientsNew = length(patientDB);

for p=1:numPatientsNew
    rows2keep = [];
    for t=1:length(patientDB(p).testdata)
        if ~isempty([patientDB(p).testdata(t).MinsToSurgery])
            rows2keep = [rows2keep, t];
        end
    end
    patientDB(p).testdata = patientDB(p).testdata(rows2keep); %keep only those tests that contain values
end

% commonTests = [patientDB(1).testdata.TestType];
% for p=2:length(patientDB)
%     [commonTests,ia,ib] = intersect(commonTests, [patientDB(p).testdata.TestType]);
%     patientDB(p).testdata = patientDB(p).testdata(ib); %keep only those tests that are common across entire dataset
% end

%Go into training script:
training