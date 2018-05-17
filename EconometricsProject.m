clear
clc
clf
format long g

%% Data
%http://tonto.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=RWTC&f=D
oil = importdata("C:\\Users\\Hansa\\Documents\\Alex's Stuff\\econometricsProject\\crude_oil_price.csv");
O = length(oil.data);
days = 504;

%daily oil spot price at close
doc = oil.data(O-days+1:O); %because excel stores most past dates first

%Represents one market day
h = 1/252;
sqrth = sqrt(h);

%% Set desired number of jumps per year
jumps = 100;
lambda = jumps*h;
fn = @(m) poissrnd(lambda,m,1); %poisson random variable with lambda = jumps/252

split = 0.5;testLength = days*split -1;
%% Calibration
%Set Y to testing data for spot prices.
%Set S to 1 day before the dates in the testing data. Treat as spot price.

Y = doc(2:days*split); %set Y
S = doc(1:testLength);

%Calculate lognormal rv and its mean and variance
lgnormal = (Y - S)./S + 1; %y ~lognormal(mu,delta^2)
log_mu = mean(log(lgnormal)); %E[ln(y)] = mu
log_var = var(log(lgnormal)); %Var[ln(y)] = delta^2
k = mean(lgnormal-1); %E[y-1] =: k
muparam = [log_mu;0]; %alpha hat MJD
dparam = [log_var;0]; %delta^2 hat MJD
kparam = [k;0]; %k hat
%Calculate drift and volatility
logReturns = log(Y)-log(S); 
Sbar = mean(logReturns); %E[ln(St/St-1)] 
S2 = (1/(testLength-1))*sum((logReturns-Sbar).^2);%var[ln(St/St-1)]
vol = sqrt(S2)/sqrt(h); % vol GBM
vol_jump = sqrt((S2-lambda*log_var)/h); %vol MJD
drift = Sbar/h+0.5*vol^2; %drift GBM
drift_jump = (Sbar-lambda*log_mu)/h+0.5*vol_jump^2+lambda*k;% drift jump

%% Plot calibrated estimates
disp("Historical sample statistics")
sampleStats = {'Z', 'EZ', 'VarZ'};
Z = ["log Returns (LR)";"Relative Returns + 1 (y)"];
averages = [log_mu;Sbar];
variances = [log_var;S2];
sampleTable = table(Z,averages,variances);
sampleTable.Properties.VariableNames = sampleStats;
disp(sampleTable);

disp(strcat("Merton Jump Diffusion calibration, jumps = ",num2str(jumps)));
MJDnames = {'lambda','mu_y','delta_y','k','alpha','sigma'};
MJD_est = table(lambda,log_mu,sqrt(log_var),k,drift_jump,vol_jump);
MJD_est.Properties.VariableNames = MJDnames;
disp(MJD_est);

disp("Geometric Brownian Motion calibration");
GBMnames = {'mu_LR','sigma_LR'};
GBM_est = table(drift,vol);
GBM_est.Properties.VariableNames = GBMnames;
disp(GBM_est);

%% Initialize constants and vectors for Monte Carlo Simulation
%Scale parameter estimates
parameters = [drift_jump-0.5*vol_jump^2-lambda*k,vol_jump;
    drift - 0.5*vol^2,vol];

%hold off
v0 = doc(days*split+1); %Todays spot price
vhat = zeros(days*split,1); %will record path for GBM
vhat(1) = v0;
vhatJump = vhat; %Will record path for GBM with jump
avgJumpErr = zeros(days*split,1); %will record MSE for GBM with jump
avgGBMerr = avgJumpErr; %will record MSE for GBM

averageJump = 0;
jumpCount = 0;
averageJumpCount = 0;

%look at latter half of data
doc = doc(split*days+1:days);

hold on

%% Forecasting with Monte Carlo Simulation
paths = 1000; %set to desired number of paths
path = 0;
m1 = "GBM with jump";m2 = "GBM";
sq_differences = zeros(paths,1);
invDays = 1/((1-split)*days-1);
while path < paths 
    %Generate sample path
    for i = 2:(1-split)*days
        nr = fn(1);
        jumpCount = jumpCount + nr;
        jump = normrnd(log_mu,log_var)*(nr>0); %jump sample
       
        dW = sqrth*normrnd(0,1); %brownian motion sample
        vhat(i) = vhat(i-1)*exp(parameters(2,1)*h + parameters(2,2)*dW); %AR(1) without CPP
        %dW = sqrth*normrnd(0,1);
        vhatJump(i) = vhatJump(i-1)*exp(parameters(1,1)*h + parameters(1,2)*dW + jump); %GBM with CPP
        
        averageJump = averageJump+jump;
    end
    
    
    %Plot GBM and MJD 
    l2= plot(vhat,'Color','red');
    l3 = plot(vhatJump,'Color','green');
    path = path + 1;
    averageJumpCount = averageJumpCount + jumpCount;
    jumpError = doc - vhatJump; 
    GBMError = doc - vhat; 
    
    %sample MSED (Mean square error difference)
    sq_differences(path) = invDays*sum(GBMError.^2-jumpError.^2);
    
    %Aggregate the errors
    avgJumpErr = avgJumpErr + jumpError.^2;
    avgGBMerr = avgGBMerr + GBMError.^2;
    
    jumpCount=0;
    jump = 0;
    
end

%% Plot Monte Carlo Results
l1 = plot(doc,'Color','blue'); 
m1 = "Actual"; m2 = "GBM";m3 = "GBM with jump";
legend([l1,l2,l3],[m1,m2,m3])
title("Monte Carlo Paths against real path")
hold off

averageJumpCount = averageJumpCount/paths;
averageJump = averageJump/averageJumpCount;
avgJumpErr = avgJumpErr*invDays;
avgGBMerr = avgGBMerr*invDays;

mean_MSE_jump = mean(avgJumpErr);
mean_MSE_GBM = mean(avgGBMerr);
var_MSE_jump = var(avgJumpErr);
var_MSE_GBM = var(avgGBMerr);

model = ["MJD";"GBM"];
mean_MSE = [mean_MSE_jump;mean_MSE_GBM];
var_MSE = [var_MSE_jump;var_MSE_GBM];
drift = [parameters(1,1);parameters(2,1)];
volatility = [parameters(1,2);parameters(2,2)];

paramTable = table(model,drift,volatility,mean_MSE,var_MSE);
disp("Parameter values based of estimates");
disp(paramTable);

figure 
l6 = bar(avgGBMerr-avgJumpErr,'green');
title("GBM Error less GBM with jump error")


%% Test statistics and P values
mean_MSED = mean(sq_differences);
std_MSED = std(sq_differences);
t_stat = sqrt(paths)*mean_MSED/std_MSED;
p_value = normpdf(t_stat,0,1);
stats = table(paths,mean_MSED,std_MSED,t_stat,p_value);
disp(stats);