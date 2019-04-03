import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator import itemgetter
import os
import argparse
from scipy.interpolate import spline
import pdb
from sklearn import metrics

parser = argparse.ArgumentParser(description="main.py")
parser.add_argument("--path", type=str, default="", help="learner method")
args = parser.parse_args()

class plots_visualize:
	def __init__(self):
		pass
	
	def visualize_sparsity(self, pid, data):
		#create sub	arr of pid only
		data_arr = np.array(data)
		match = [x for x in list(data_arr) if x[0] == pid]
		match_arr = np.array(match)
		x_time = (match_arr[:,3]/1440)/60 #convert min to day
		y_test = match_arr[:,1]
		return x_time, y_test
	
	#plot: visualize data sparsity
	#To change: ntopTests = 811, discard_testRelRange self.xnew = self.x1new
	def data_sparsity(self, xy):
		pid = xy.iloc[1000][0] #pick any pid [1000][0] no_inf, [3000][0] yes_inf
		x_time, y_test = self.visualize_sparsity(pid, xy)
		plt.gca().invert_xaxis()
		plt.scatter(x_time, y_test, color='b')
		plt.xlabel('Surgery Date - Blood Test Date (Days)')
		plt.ylabel('Available Test Index')
		plt.grid()
		plt.savefig('DataSparsity1')
		plt.close()

	#plot: imputation rate
	def impute_rate(self):
		impute_PID = np.load(os.path.join(args.path,'LOCF_keep5Test_count_imputePID.npy'))
		PID = range(0, len(impute_PID))
		#plt.plot(PID, impute_PID)
		plt.scatter(PID, impute_PID)
		plt.invert_xaxis()
		plt.xlabel("patient index")
		plt.ylabel("number of imputations")
		plt.grid()
		plt.savefig('num_LOCF_kepp5Test')
		plt.close()
		
	#plot: visualize before & after imputation
	def before_after_impute(self, xy):
		pid = xy.iloc[0][0] #[0]=inf, [5]=no_inf
		testNum = 0
		arr_xy = np.array(xy)
		match = [x for x in list(arr_xy) if x[0] == pid] #get pid
		match_test = [x for x in list(match) if x[1] == testNum]
		match_test = np.array(sorted(match_test, key=itemgetter(3), reverse=True))
		before_imputeX = match_test[:,3]/1440 #bTest time
		before_imputeY = match_test[:,2] #bTest val
		
		after_impute = np.load(os.path.join(args.path,'NN_20tests_10samples.npy'))
		pid_idx = np.where(after_impute[:,0] == pid)[0][0]
		matchA = after_impute[pid_idx]
		match_testA = matchA[testNum] #already time sorted
		after_imputeX = match_testA[:,3]/1440
		after_imputeY = match_testA[:,2]
		
		plt.scatter(before_imputeX, before_imputeY, color='blue')
		plt.plot(before_imputeX, before_imputeY, 'b-')
		#before_imputeXnew = np.linspace(before_imputeX.min(), before_imputeX.max(), 3000)
		#plt_smooth = spline(before_imputeX.astype(float), before_imputeY.astype(float), before_imputeXnew)
		plt.scatter(after_imputeX, after_imputeY, color='red')
		plt.plot(after_imputeX, after_imputeY, color='red')
		
		pdb.set_trace()
		#second pid
		pid = xy.iloc[5][0]
		arr_xy = np.array(xy)
		match = [x for x in list(arr_xy) if x[0] == pid] #get pid
		match_test = [x for x in list(match) if x[1] == testNum]
		match_test = np.array(sorted(match_test, key=itemgetter(3), reverse=True))
		before_imputeX = match_test[:,3]/1440 #bTest time
		before_imputeY = match_test[:,2] #bTest val
		
		after_impute = np.load(os.path.join(args.path,'NN_20tests_10samples.npy'))
		pid_idx = np.where(after_impute[:,0] == pid)[0][0]
		matchA = after_impute[pid_idx]
		match_testA = matchA[testNum] #already time sorted
		after_imputeX = match_testA[:,3]/1440
		after_imputeY = match_testA[:,2]
		
		plt.scatter(before_imputeX, before_imputeY, color='black')
		plt.plot(before_imputeX, before_imputeY, 'k-')
		plt.scatter(after_imputeX, after_imputeY, color='green')
		plt.plot(after_imputeX, after_imputeY, color='green')
		
		#plt.title("before_after imputation")
		plt.ylabel("Non-Standardized Blood Test Results")
		plt.xlabel("Surgery Date - Blood Test Date (Days)")
		plt.legend(['Before impute non-SSI', 'After impute non-SSI', 'Before impute SSI', 'After impute SSI'], loc='best')
		plt.grid()
		plt.savefig('NN_before_after_impute')
		plt.close()

	#plot: ROC curve
	def roc_curveA(self, ytestnn, probLRnn, ytestlocf, probLRlocf, ytestgp, probLRgp, probMLPnn, probMLPlocf, probMLPgp):
			
		fprLRlocf, tprLRlocf, _ = metrics.roc_curve(ytestlocf, probLRlocf)
		aucLRlocf = metrics.roc_auc_score(ytestlocf, probLRlocf)
		plt.plot(fprLRlocf, tprLRlocf, label='$LR_{nn}$ AUC = %0.3f' % aucLRlocf, color='blue')
		
		fprLRnn, tprLRnn, _ = metrics.roc_curve(ytestnn, probLRnn)
		aucLRnn = metrics.roc_auc_score(ytestnn, probLRnn)
		plt.plot(fprLRnn, tprLRnn, label='$LR_{locf}$ AUC = %0.3f' % aucLRnn, color='black')	
		
		fprLRgp, tprLRgp, _ = metrics.roc_curve(ytestgp, probLRgp)
		aucLRgp = metrics.roc_auc_score(ytestgp, probLRgp)
		plt.plot(fprLRgp, tprLRgp, label='$LR_{gp}$ AUC = %0.3f' % aucLRgp, color='magenta')
			
		fprMLPnn, tprMLPnn, _ = metrics.roc_curve(ytestnn, probMLPnn)
		aucMLPnn = metrics.roc_auc_score(ytestnn, probMLPnn)
		plt.plot(fprMLPnn, tprMLPnn, label='$MLP_{nn}$ AUC = %0.3f' % aucMLPnn, color='green')
		
		fprMLPlocf, tprMLPlocf, _ = metrics.roc_curve(ytestlocf, probMLPlocf)
		aucMLPlocf = metrics.roc_auc_score(ytestlocf, probMLPlocf)
		plt.plot(fprMLPlocf, tprMLPlocf, label='$MLP_{locf}$ AUC = %0.3f' % aucMLPlocf, color='red')
		
		fprMLPgp, tprMLPgp, _ = metrics.roc_curve(ytestgp, probMLPgp)
		aucMLPgp = metrics.roc_auc_score(ytestgp, probMLPgp)
		plt.plot(fprMLPgp, tprMLPgp, label='$MLP_{gp}$ AUC = %0.3f' % aucMLPgp, color='cyan')
				
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.grid()
		plt.legend(loc=4)
		plt.savefig('ROCa')
		plt.close()
	
	#plot: ROC curve
	def roc_curveB(self, ytestLSTM, probLSTM, ytestGP_LR, probGP_LR, ytestGP_MLP, probGP_MLP):
		
		fprLSTM, tprLSTM, th = metrics.roc_curve(ytestLSTM, probLSTM)
		aucLSTM = metrics.roc_auc_score(ytestLSTM, probLSTM)
		plt.plot(fprLSTM, tprLSTM, label='LSTM AUC = %0.3f' % aucLSTM, color='blue')
		
		fprGP_LR, tprGP_LR, th = metrics.roc_curve(ytestGP_LR, probGP_LR)
		aucGP_LR = metrics.roc_auc_score(ytestGP_LR, probGP_LR)
		plt.plot(fprGP_LR, tprGP_LR, label='$GP_{LR}$ AUC = %0.3f' % aucGP_LR, color='black')
		
		fprGP_MLP, tprGP_MLP, th = metrics.roc_curve(ytestGP_MLP, probGP_MLP)
		aucGP_MLP = metrics.roc_auc_score(ytestGP_MLP, probGP_MLP)
		plt.plot(fprGP_MLP, tprGP_MLP, label='$GP_{MLP}$ AUC = %0.3f' % aucGP_MLP, color='red')
		
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.grid()
		plt.legend(loc=4)
		plt.savefig('ROCb')
		plt.close()
	
	#plot: Cost curve
	def cost_curveA(self, cmLRnn, cmLRlocf, cmLRgp, cmMLPnn, cmMLPlocf, cmMLPgp):
		#cost matrix:
		#        Predict yes    Predict no
		#Actual yes  -1            100
		#Actual no   1              0
		costm = np.array([[-1,100],[1,0]])
		fncost = costm[0,1]
		fpcost = costm[1,0]
		
		fnLRnn = cmLRnn[0,1]/(cmLRnn[0,1]+cmLRnn[1,1])
		fpLRnn = cmLRnn[1,0]/(cmLRnn[0,0]+cmLRnn[1,0])
		
		fnLRlocf = cmLRlocf[0,1]/(cmLRlocf[0,1]+cmLRlocf[1,1])
		fpLRlocf = cmLRlocf[1,0]/(cmLRlocf[0,0]+cmLRlocf[1,0])
		
		fnLRgp = cmLRgp[0,1]/(cmLRgp[0,1]+cmLRgp[1,1])
		fpLRgp = cmLRgp[1,0]/(cmLRgp[0,0]+cmLRgp[1,0])
		
		fnMLPnn = cmMLPnn[0,1]/(cmMLPnn[0,1]+cmMLPnn[1,1])
		fpMLPnn = cmMLPnn[1,0]/(cmMLPnn[0,0]+cmMLPnn[1,0])
		
		fnMLPlocf = cmMLPlocf[0,1]/(cmMLPlocf[0,1]+cmMLPlocf[1,1])
		fpMLPlocf = cmMLPlocf[1,0]/(cmMLPlocf[0,0]+cmMLPlocf[1,0])
		
		fnMLPgp = cmMLPgp[0,1]/(cmMLPgp[0,1]+cmMLPgp[1,1])
		fpMLPgp = cmMLPgp[1,0]/(cmMLPgp[0,0]+cmMLPgp[1,0])
		
		probPos = np.linspace(0,1,11)
		X = np.zeros((len(probPos)))
		for i in range(len(probPos)):
			X[i] = (probPos[i] * fncost) / (probPos[i]*fncost + (1-probPos[i]*fpcost))
		yLRnn = fnLRnn * X + fpLRnn * (1-X)
		yLRlocf = fnLRlocf * X + fpLRlocf * (1-X)
		yLRgp = fnLRgp * X + fpLRgp * (1-X)
						
		yMLPnn = fnMLPnn * X + fpMLPnn * (1-X)
		yMLPlocf = fnMLPlocf * X + fpMLPlocf * (1-X)
		yMLPgp = fnMLPgp * X + fpMLPgp * (1-X)				
		#yLSTM = fnLSTM * X + fpLSTM * (1-X)
				
		plt.plot(X, yLRnn, label='$LR_{nn}$', color='blue')
		plt.plot(X, yLRlocf, label='$LR_{locf}$', color='black')
		plt.plot(X, yLRgp, label='$LR_{gp}$', color='magenta')
		plt.plot(X, yMLPnn, label='$MLP_{nn}$', color='green')
		plt.plot(X, yMLPlocf, label='$MLP_{locf}$', color='red')
		plt.plot(X, yMLPgp, label='$MLP_{gp}$', color='cyan')
		plt.legend(loc=0)
		plt.grid()
		plt.xlabel('Probability of a Positive Example P(+)')
		plt.ylabel('Error Rate')
		plt.savefig('CostCurveA')
		plt.close()
		
	#plot: Cost curve
	def cost_curveB(self, cmLSTM, cmGPlr, cmGPmlp):
		#cost matrix:
		#        Predict yes    Predict no
		#Actual yes  -1            100
		#Actual no   1              0
		costm = np.array([[-1,100],[1,0]])
		fncost = costm[0,1]
		fpcost = costm[1,0]
			
		fnLSTM = cmLSTM[0,1]/(cmLSTM[0,1]+cmLSTM[1,1])
		fpLSTM = cmLSTM[1,0]/(cmLSTM[0,0]+cmLSTM[1,0])

		fnGPlr = cmGPlr[0,1]/(cmGPlr[0,1]+cmGPlr[1,1])
		fpGPlr = cmGPlr[1,0]/(cmGPlr[0,0]+cmGPlr[1,0])
		
		fnGPmlp = cmGPmlp[0,1]/(cmGPmlp[0,1]+cmGPmlp[1,1])
		fpGPmlp = cmGPmlp[1,0]/(cmGPmlp[0,0]+cmGPmlp[1,0])
		
		probPos = np.linspace(0,1,11)
		X = np.zeros((len(probPos)))
		for i in range(len(probPos)):
			X[i] = (probPos[i] * fncost) / (probPos[i]*fncost + (1-probPos[i]*fpcost))
		yLSTM = fnLSTM * X + fpLSTM * (1-X)
		yGPlr = fnGPlr * X + fpGPlr * (1-X)
		yGPmlp = fnGPmlp * X + fpGPmlp * (1-X)
										
		plt.plot(X, yLSTM, label='LSTM', color='blue')
		plt.plot(X, yGPlr, label='$GP_{LR}$', color='black')
		plt.plot(X, yGPmlp, label='$GP_{MLP}$', color='red')
		plt.legend(loc=0)
		plt.grid()
		plt.xlabel('Probability of a Positive Example P(+)')
		plt.ylabel('Error Rate')
		plt.savefig('CostCurveB')
		plt.close()

################################################################################
#Plot: learning curve
'''Xtrain, ytrain = shuffle(Xtrain, ytrain, random_state=42)
ytrain = ytrain.ravel()
train_sizes, train_scores, test_scores = learning_curve(LogisticRegression(), Xtrain, ytrain, cv=10, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 15))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, '--', color='k', label='Training score')
plt.plot(train_sizes, test_mean, color='b', label='Cross-validation score')
plt.xlabel("Training set size")
plt.ylabel("Accuracy score")
plt.legend(loc="best")
plt.tight_layout()
plt.grid()
plt.savefig('LR_LearningCurve')
plt.close()'''


'''LR_learningCurve_trainErr = np.array([0.21, 0.189, 0.172, 0.151])
LR_learningCurve_testErr = np.array([0.237, 0.226, 0.248, 0.233])
LR_learningCurve_samples = np.array([200, 400, 600, 800])
plt.scatter(LR_learningCurve_samples, LR_learningCurve_trainErr, color='b')
plt.plot(LR_learningCurve_samples, LR_learningCurve_trainErr, 'b-')
plt.scatter(LR_learningCurve_samples, LR_learningCurve_testErr, color='k')
plt.plot(LR_learningCurve_samples, LR_learningCurve_testErr, 'k-')
plt.xlabel('Number of samples')
plt.ylabel('Error')
plt.grid()
plt.savefig('LR_LearningCurve')
plt.close()'''

#Plot: epoch vs. accuracy
'''plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(['train', 'validation'], loc='upper left')
plt.grid()
plt.savefig('MLP_MEANmodel_acc')
plt.close()

#Plot: epoch vs. loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(['train', 'validation'], loc='upper left')
plt.grid()
plt.savefig('MLP_MEANmodel_loss')
plt.close()'''

'''
#Plot: PR curve
precision, recall, th = metrics.precision_recall_curve(ytest, ypred)
pr_auc = metrics.auc(recall, precision)
plt.plot(recall, precision)
plt.legend(['AUC = %0.2f' % pr_auc], loc='upper left')
plt.xlabel('recall')
plt.ylabel('precision')
plt.grid()
plt.savefig('MLP_LOCFmodel_pr')
plt.close()'''
	
