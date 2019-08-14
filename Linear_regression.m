% Using linear regression method to predict labels
% Compare predict results by one versus others method

% Prepare ramdonalized input data
load('D_iris.mat')
D = D_iris(1:4,:);
x1 = D(:,1:50);
x2 = D(:,51:100);
x3 = D(:,101:150);
rand('state',111);
r1 = randperm(50);
xtr1 = x1(:,r1(1:40));
xte1 = x1(:,r1(41:50));
rand('state',112);
r2 = randperm(50);
xtr2 = x2(:,r2(1:40));
xte2 = x2(:,r2(41:50));
rand('state',113);
r3 = randperm(50);
xtr3 = x3(:,r3(1:40));
xte3 = x3(:,r3(41:50));

Dtrain_1 = [xtr1 xtr2 xtr3];  % Prepare trainning data for 3 classes
Dtrain_2 = [xtr2 xtr3 xtr1];
Dtrain_3 = [xtr3 xtr1 xtr2];
Dtest = [xte1 xte2 xte3];

% Train dataset xtr1 as class P, xtr2&xtr3 as class N
y = [ones(40,1);-ones(80,1)];
x_hat = ones(120,5);
x_hat(1:120,1:4) = Dtrain_1';
w_hat = inv(x_hat'*x_hat)*x_hat'*y;
w = w_hat';
w_output = w(:,1:4);    % Obtain values of w from 1-4 col
b = w(:,5);             % Obatain value of bias from last col of w

%Train dataset xtr2 as class P, xtr3&xtr1 as class N
x_hat2 = ones(120,5);
x_hat2(1:120,1:4) = Dtrain_2';
w_hat2 = inv(x_hat2'*x_hat2)*x_hat2'*y;
w2 = w_hat2';
w_output2 = w2(:,1:4);
b2 = w2(:,5);

%Train dataset xtr3 as class P, xtr1&xtr2 as class N
x_hat3 = ones(120,5);
x_hat3(1:120,1:4) = Dtrain_3';
w_hat3 = inv(x_hat3'*x_hat3)*x_hat3'*y;
w3 = w_hat3';
w_output3 = w3(:,1:4);
b3 = w3(:,5);

% Test dataset
y_values = w_output*Dtest+b*ones(1,30);
y_values2 = w_output2*Dtest+b2*ones(1,30);
y_values3 = w_output3*Dtest+b3*ones(1,30);
% u = sign(values);

% Assign labels of 3 classes
y_k = zeros(30,3);
[row, col] = size(y_k);
for i=1:row
    if y_values(1,i)>y_values2(1,i)&&y_values(1,i)>y_values3(1,i)
        y_k(i,1) = 1;
    elseif y_values2(1,i)>y_values(1,i)&&y_values2(1,i)>y_values3(1,i)
        y_k(i,2) = 1;
    else 
        y_k(i,3)=1;
    end
end

% Calculate miss classified labels
E = y_k';
miss_class = 0;
test_class1 = y_k(1:10,1);
test_class2 = y_k(11:20,2);
test_class3 = y_k(21:30,3);
for i = 1:10
    if test_class1(i,1)==0||test_class2(i,1)==0||test_class3(i,1)==0
        miss_class = miss_class + 1;
    end
end

accuracy = (row-miss_class)/row % Display accuracy
miss_class % Display number of miss classified prediction
