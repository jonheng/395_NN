% Task 8
[clsfError_train1,clsfError_val1,L_train1,L_val1,stats1] = task2NNfortask8Function();
[clsfError_train2,clsfError_val2,L_train2,L_val2,stats2,nn] = task8NNFunction();

f=figure();
x_axis1=find(clsfError_train1);
x_axis2=find(clsfError_train2);
plot(x_axis1,clsfError_train1(x_axis1),x_axis1,clsfError_val1(x_axis1),x_axis2,clsfError_train2(x_axis2),x_axis2,clsfError_val2(x_axis2));
legend('Initial training error','Initial validation error','Optimal training error','Optimal validation error');
title('Classification errors for neural networks trained on initial vs optimal set of parameters');
xlabel('Number of Epochs');
ylabel('Classification Error');
saveas(f,'figures/clsfError8.png');
plot(x_axis1,L_train1(x_axis1),x_axis1,L_val1(x_axis1),x_axis2,L_train2(x_axis2),x_axis2,L_val2(x_axis2));
legend('Iniital training loss','Initial validation loss','Optimal training loss','Optimal validation loss');
title('Loss values for neural networks trained on initial vs optimal set of parameters');
xlabel('Number of Epochs');
ylabel('Loss');
saveas(f,'figures/loss8.png');
close all
save('optimalNN.mat','nn');
