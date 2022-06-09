clear all;
% x(t+1) = Ax(t) + w
% y(t) = Bx(t) + v
% w ~ N(0, Q)
% v ~ N(0, R)

%% Parameters
A = 1.001; % transition matrix
B = 108; % emission matrix
Q = 0.25; % covariance of noise w
R = 0.3; % covariance of noise v
x0 = 0; % initial state
len = 200; % length of the sequence

%% Initialization
x_true = zeros(1, len);
x_pred = zeros(1, len);
y = zeros(1, len);
x_true(1) = x0;
x_pred(1) = x0;
seed = 7;

%% Preparation of true location
randn('state', seed);
y(1) = B * x_true(1) + normrnd(0, R^0.5);

for i = 2:len
    x_true(i) = A * x_true(i - 1) + normrnd(0, Q^0.5);
    y(i) = B * x_true(i) + normrnd(0, R^0.5);
end

%% Prediction
P = 0;

for i = 2:len
    % X(t+1|t) update variable
    x_pred(i) = A * x_pred(i - 1);
    % P(t+1|t) predicted covariance matrix
    P = A * P * A' + Q;
    % K(t+1) Kalman gain
    K = P * B * (B * P * B' + R)';
    % X(t+1|t+1) estimated variable
    x_pred(i) = K * y(i) + (1 - K * B) * x_pred(i);
    % P(t+1|t+1) estimated covariance matrix
    P = (1 - K * B) * P;
end

%% Evaluation
mse = sum((x_pred - x_true).^2) ./ size(x_true, 2);
disp(['MSE:', num2str(mse)]);

%% Illustration
plot(x_pred, '-', 'linewidth', 1.5)
hold on
plot(x_true, '--', 'linewidth', 1.5)
xlabel('time step');
ylabel('x');
legend('prediction of x', 'true x');
grid on;
title(['Kalman Filter (x0=', num2str(x0), ' A=', num2str(A), ')']);
saveas(gcf, 'Kalman.png');
