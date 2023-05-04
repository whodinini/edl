%This code implements the physiological constraints hypothesis for the emergence of DoL in newly formed groups
%The "physiological constraints hypothesis" proposes that DoL emerges to 
%manage physiological costs associated with task performance.
%According to this idea, tasks that require high physiological investment,
%such as foraging or brood care, can be costly to perform continuously, 
%creating a trade-off between task performance and other physiological processes
%To manage these physiological costs, individuals may adjust their behavior
%in ways that allow them to optimize their physiology for a particular task, 
%such as resting. These adjustments can create a division of labor,
%where individuals specialize in different tasks to maximize their overall fitness.
%We argue that physiological benefits of specalization (i.e., 'concentrating on a subset of tasks')
%could lead group members to engage in task partitioning, which provides mutual benefits
%Moroever, the fitness advantage of behavioral variation could serve as a stepping stone to the evolution of DOL
%Unlike conventional models of DoL which focus on ontogeny of behavior (i.e., response-thresholds),
%here, we focus on the physiology of behavior within newly-formed groups.

clear all
%%%Colors
gold = [0.9290, 0.6940, 0.1250];
purple = [0.4940, 0.1840, 0.5560];
orange = [0.8500, 0.3250, 0.0980];
red = [0.6350, 0.0780, 0.1840];
green = [0.4660, 0.6740, 0.1880];
blue = [0 0 0.6350];
black = [0, 0, 0];
grey = black+0.75;

%INITIALIZATION
%clear all;
T = 1000; %time when reproduction occurs
tspan = 1:T; %simulation period (i.e., the morning of day 1 to morning of day 100)

A = zeros(length(tspan)-1,1);
B = zeros(length(tspan)-1,1);
X1 = zeros(length(tspan)-1,1);
X2 = zeros(length(tspan)-1,1);
X0 = zeros(length(tspan)-1,1);
Y1 = zeros(length(tspan)-1,1);
Y2 = zeros(length(tspan)-1,1);
Y0 = zeros(length(tspan)-1,1);
actionX = zeros(length(tspan)-1,1);
actionY = zeros(length(tspan)-1,1);

D=[]; %dol matrix to be populated
Dtime = [];

%parameters
n=2; %number of individuals
m=2; %number of tasks
beta=1; %somatic recovery rate
phi = 2; %metabolic operating cost (test values: 2, 0.5, 1, 5, 10, 20)
r =25; % fatigue rate (test values: 0.5, 1, 5, 10, 20)
gamma = 0.5; %pratice efficiency (task learning rate)
omega = 0.5; %fatigue cost (task forgetting rate)
alpha = 0.1; %task performance efficiency
kappa = 1; %shape parameter for S(x)
%S = @(x) exp(-kappa*x); %metabolic improvement function, S(x). Note: x is a dummy variable
S = @(x) 1./(kappa*x);
delta=0.0; %task demand rate
eta=0.25; %quitting prob.
etad=1; %adaptive decision prob.
niter = 5000; %number of interations

model=1;  %1:self-maintenance, 2:random, 3:response-threshold, 4:task fidelity
block_one_agent=0; %for single-agent model, set equals to 1

TWmat = []; TEmat = [];

%xvalues= linspace(0,50,11); %r
xvalues= linspace(0,1,11); %beta
yvalues = linspace(0,1,11); %gamma


do_heatmaps=1;

if do_heatmaps==1
    for ind1 = 1:length(xvalues)
        rho1 = xvalues(ind1);  %vary beta - increasing down rows
        if rho1==0.5
            disp('r==0.5')
        end
        beta = rho1;
        for ind2 = 1:length(yvalues)
            rho2 = yvalues(ind2); %vary gamma - increasing across columns
            gamma = rho2;
            
            
            for iter=1:niter
                
                A = zeros(length(tspan)-1,1);
                B = zeros(length(tspan)-1,1);
                X1 = zeros(length(tspan)-1,1);
                X2 = zeros(length(tspan)-1,1);
                X0 = zeros(length(tspan)-1,1);
                Y1 = zeros(length(tspan)-1,1);
                Y2 = zeros(length(tspan)-1,1);
                Y0 = zeros(length(tspan)-1,1);
                actionX = zeros(length(tspan)-1,1);
                actionY = zeros(length(tspan)-1,1);
                
                %initial conditions
                A(1) = 50; %amount of work remaining for task 1 on day 1
                B(1) = 50; %amount of work remaining for task 2 on day 1
                
                X1(1) = 1;   %competence of agent X in task 1 on day 1
                X2(1) = 1;   %competence of agent X in task 2 on day 1
                X0(1) = 1;   %fatigue level on day 1
                
                Y1(1) = 1;   %competence of agent Y in task 1 on day 1
                Y2(1) = 1;   %competence of agent Y in task 2 on day 1
                Y0(1) = 1;   %fatigue level on day 1
                
                countX1 = 1; %number of task 1 reps by agent X on day 1
                countX2 = 1; %number of task 2 reps by agent X on day 1
                
                countY1 = 1; %number of task 1 reps by agent Y on day 1
                countY2 = 1; %number of task 2 reps by agent Y on day 1
                
                ActivityMat = [countX1, countX2; countY1, countY2];
                dol_metric = computeDOL(ActivityMat); %calls an external funciton
                Dtime(1) = dol_metric;
                
                %Assume there's an asymmetry in agents's starting behavior - this is an act of nature
                actionX(1) = 1; % agent X's behavioral state on day 1
                actionY(1) = 2; % agent Y's behavioral state on day 1
                
                %                 %Randomize agents statrting action
                %                 if rand(1) < 0.5 %the starting prob.
                %                 actionX(1) = 1; % agent X's behavioral state on day 1
                %                 actionY(1) = 2; % agent Y's behavioral state on day 1
                %                 else
                %                 actionX(1) = 0; % agent X's behavioral state on day 1
                %                 actionY(1) = 0; % agent Y's behavioral state on day 1
                %                 end
                
                if block_one_agent==1
                    actionY(1)=0;
                end
                
                lasttaskX=actionX(1); lasttaskY=actionY(1);
                
                for i = 1:length(tspan)-1
                    if actionX(i) == 0 && actionY(i) == 0
                        %update tasks
                        A(i+1) = A(i);
                        B(i+1) = B(i);
                        
                        %update fatigue for agent X
                        test = X0(i) - beta*(r); %calculate fatigue decrement for resting
                        if test > 1
                            X0(i+1) = test;
                        else
                            X0(i+1) = 1; %set floor at 1
                        end
                        %update fatigue for agent Y
                        test = Y0(i) - beta*(r); %calculate fatigue decrement for resting
                        if test > 1
                            Y0(i+1) = test;
                        else
                            Y0(i+1) = 1; %set floor at 1
                        end
                    end
                    
                    if actionX(i) == 0 && actionY(i) == 1
                        %update tasks
                        countY1 = countY1+1;
                        test = A(i) - alpha;
                        if test > 1
                            A(i+1) = test;
                        else
                            A(i+1) = 1; %set floor at 1
                        end
                        B(i+1) = B(i);
                        %update fatigue for agent X
                        test = X0(i) - beta*(r); %calculate fatigue decrement for resting
                        if test > 1
                            X0(i+1) = test;
                        else
                            X0(i+1) = 1; %set floor at 1
                        end
                        %update fatigue for agent Y
                        Y0(i+1) = Y0(i) + r*S(Y1(i)); %calculate fatigue increment for agent Y
                    end
                    
                    if actionX(i) == 0 && actionY(i) == 2
                        %update tasks
                        countY2 = countY2 +1;
                        test = B(i) - alpha;
                        if test > 1
                            B(i+1) = test;
                        else
                            B(i+1) = 1; %set floor at 1
                        end
                        A(i+1) = A(i);
                        %update fatigue for agent X
                        test = X0(i) - beta*(r); %calculate fatigue decrement for resting
                        if test > 1
                            X0(i+1) = test;
                        else
                            X0(i+1) = 1; %set floor at 1
                        end
                        %update fatigue for agent Y
                        Y0(i+1) = Y0(i) + r*S(Y2(i)); %calculate fatigue increment for agent Y
                    end
                    
                    if actionX(i) == 1 && actionY(i) == 0
                        %update tasks
                        countX1 = countX1 +1;
                        test = A(i) - alpha;
                        if test > 1
                            A(i+1) = test;
                        else
                            A(i+1) = 1; %set floor at 1
                        end
                        B(i+1) = B(i);
                        %update fatigue for agent X
                        X0(i+1) = X0(i) + r*S(X1(i)); %calculate fatigue increment for agent X
                        %update fatigue for agent Y
                        test = Y0(i) - beta*(r); %calculate fatigue decrement for resting
                        if test > 1
                            Y0(i+1) = test;
                        else
                            Y0(i+1) = 1; %set floor at 1
                        end
                    end
                    
                    if actionX(i) == 1 && actionY(i) == 1
                        %update tasks
                        countX1 = countX1 +1; countY1 = countY1+1;
                        test = A(i) - 2*alpha;
                        if test > 1
                            A(i+1) = test;
                        else
                            A(i+1) = 1; %set floor at 1
                        end
                        B(i+1) = B(i);
                        %update fatigue
                        X0(i+1) = X0(i) + r*S(X1(i)); %calculate fatigue increment for agent X
                        Y0(i+1) = Y0(i) + r*S(Y1(i)); %calculate fatigue increment for agent Y
                    end
                    
                    if actionX(i) == 1 && actionY(i) == 2
                        %update tasks
                        countX1 = countX1 +1; countY2 = countY2 +1;
                        test = A(i) - alpha;
                        if test > 1
                            A(i+1) = test;
                        else
                            A(i+1) = 1; %set floor at 1
                        end
                        test = B(i) - alpha;
                        if test > 1
                            B(i+1) = test;
                        else
                            B(i+1) = 1; %set floor at 1
                        end
                        %update fatigue
                        X0(i+1) = X0(i) + r*S(X1(i)); %calculate fatigue increment for agent X
                        Y0(i+1) = Y0(i) + r*S(Y2(i)); %calculate fatigue increment for agent Y
                    end
                    
                    if actionX(i) == 2 && actionY(i) == 0
                        %update tasks
                        countX2 = countX2 +1;
                        test = B(i) - alpha;
                        if test > 1
                            B(i+1) = test;
                        else
                            B(i+1) = 1; %set floor at 1
                        end
                        A(i+1) = A(i);
                        %update fatigue for agent X
                        X0(i+1) = X0(i) + r*S(X2(i));
                        %update fatigue for agent Y
                        test = Y0(i) - beta*(r); %calculate fatigue decrement for resting
                        if test > 1
                            Y0(i+1) = test;
                        else
                            Y0(i+1) = 1; %set floor at 1
                        end
                    end
                    
                    if actionX(i) == 2 && actionY(i) == 1
                        %update tasks
                        countX2 = countX2 +1; countY1 = countY1 +1;
                        test = A(i)  - alpha;
                        if test > 1
                            A(i+1) = test;
                        else
                            A(i+1) = 1; %set floor at 1
                        end
                        test = B(i) - alpha;
                        if test > 1
                            B(i+1) = test;
                        else
                            B(i+1) = 1; %set floor at 1
                        end
                        %update fatigue
                        X0(i+1) = X0(i) + r*S(X2(i)); %calculate fatigue increment for agent X
                        Y0(i+1) = Y0(i) + r*S(Y1(i)); %calculate fatigue increment for agent Y
                    end
                    
                    if actionX(i) == 2 && actionY(i) == 2
                        %update tasks
                        countX2 = countX2 +1; countY2 = countY2 +1;
                        test = B(i) - 2*alpha;
                        if test > 1
                            B(i+1) = test;
                        else
                            B(i+1) = 1; %set floor at 1
                        end
                        A(i+1) = A(i);
                        %update fatigue
                        X0(i+1) = X0(i) + r*S(X2(i)); %calculate fatigue increment for agent X
                        Y0(i+1) = Y0(i) + r*S(Y2(i)); %calculate fatigue increment for agent Y
                    end
                    
                    
                    %next, update competence for agent X
                    test = X1(i) + gamma*countX1 - omega*X0(i);
                    if test > 1
                        X1(i+1) = test;
                    else
                        X1(i+1) = 1; %set floor at 1
                    end
                    
                    test = X2(i) + gamma*countX2 - omega*X0(i);
                    if test > 1
                        X2(i+1) = test;
                    else
                        X2(i+1) = 1; %set floor at 1
                    end
                    
                    %next, update competence for agent Y
                    
                    %         if i<50
                    %             % holdCy1=countY1; holdCy2=countY2;
                    %             countY1=0; countY2=0;
                    %         end
                    
                    %             if block_one_agent==1
                    %                 % block competence gain of 2nd agent - this leads 1st agent
                    %                 % effectibely working alone
                    %                 countY1=0; countY2=0;
                    %             end
                    
                    test = Y1(i) + gamma*countY1 - omega*Y0(i);
                    if test > 1
                        Y1(i+1) = test;
                    else
                        Y1(i+1) = 1; %set floor at 1
                    end
                    
                    test = Y2(i) + gamma*countY2 - omega*Y0(i);
                    if test > 1
                        Y2(i+1) = test;
                    else
                        Y2(i+1) = 1; %set floor at 1
                    end
                    
                    %next calcluate decision for next period
                    %We assume agents behavior is determined a balance adaptation and memory loss
                    %The parameter eta controls the balance.
                    %If eta=0, agents always select action that maximizes their reinforcement (utlity)
                    %If eta=1, agents select select action randomly without reference to reinforcement
                    %If 0<eta<1, agents behavior is a caused mix of deterministic and stochastic processes
                    
                    %Normalized task stimuli
                    nX = phi; nY=1*phi;
                    nA =A(i);%A(i)/(A(i) + B(i));
                    nB =B(i);%B(i)/(A(i) + B(i));
                    
                    if model==1
                        if rand < etad
                            %Agent X (model 1: adaptive selection)
                            utility = 3 + X0(i)*nX + X1(i)*nA + X2(i)*nB;
                            p0 = (X0(i)*nX)/utility; %utility share for rest
                            p1 = (X1(i)*nA)/utility; %utility share for task 1
                            p2 = (X2(i)*nB)/utility; %utility share for task 2
                            weights = [p0 p1 p2];
                            %weights = circshift(weights,[1 -randi(3)]);
                            [~,mid] = max(weights); %causal model - find the action that gives the maximum utility share
                            %shift domain to [0,2]
                            mid = mid-1;
                        else
                            mid=randi(3);
                            %shift domain to [0,2]
                            mid = mid-1;
                        end
                    end
                    
                    if model ==2
                        %Model 2: Random selection - null
                        mid=randi(3);
                        %shift domain to [0,2]
                        mid = mid-1;
                    end
                    
                    if model==3
                        %Model 3: Response threshold
                        ack= randi(2); %done=0;
                        if ack==1
                            if rand < A(i)/(R(1,1)+A(i)) %&& done==0 A(i) >= R(1,1)
                                actionX(i+1) = 1;
                                %done=1;
                            elseif rand < B(i)/(R(1,2)+B(i)) % && done==0 B(i) >= R(1,2)
                                actionX(i+1) = 2;
                                %done=1;
                            else
                                actionX(i+1) = 0;
                                %done=1;
                            end
                        elseif ack==2
                            if  rand < B(i)/(R(1,2)+B(i)) % && done==0 B(i) >= R(1,2)
                                actionX(i+1) = 2;
                                %done=1;
                            elseif rand < A(i)/(R(1,1)+A(i)) %&& done==0 A(i) >= R(1,1)
                                actionX(i+1) = 1;
                                %done=1;
                            else
                                actionX(i+1) = 0;
                                %done=1;
                            end
                        end
                    end
                    
                    if model==4
                        %Model 4: Random selection - alternating
                        mid=randi(3);
                        %shift domain to [0,2]
                        mid = mid-1;
                        if mid>0
                            mid=lasttaskX;
                        end
                    end
                    
                    %if you are currently active in period i...
                    if model ~= 3 && actionX(i) > 0
                        if rand(1) < eta %the quitting prob.
                            actionX(i+1) = 0; %inactive in following period
                        else
                            actionX(i+1) = actionX(i); %else stay in current task
                        end
                        lasttaskX = actionX(i);
                    end
                    
                    %if you are currently inactive in period i...
                    if model ~= 3 && actionX(i) == 0
                        actionX(i+1) = mid; %select maximum action
                    end
                    
                    
                    
                    %Agent Y
                    if model==1
                        if rand < etad
                            %Agent Y (model 1: adaptive selection)
                            utility = 3 + Y0(i)*nY + Y1(i)*nA + Y2(i)*nB;
                            p0 = (Y0(i)*nY)/utility;
                            p1 = (Y1(i)*nA)/utility;
                            p2 = (Y2(i)*nB)/utility;
                            weights = [p0 p1 p2];
                            %weights = circshift(weights,[1 -randi(3)]);
                            [mx,mid] = max(weights);
                            %shift domain to [0,2]
                            mid = mid-1;
                        else
                            mid=randi(3);
                            %shift domain to [0,2]
                            mid = mid-1;
                        end
                    end
                    
                    if model==2
                        %Model 2: Random selection - null
                        mid=randi(3);
                        %shift domain to [0,2]
                        mid = mid-1;
                    end
                    
                    if model==3
                        %Model 3: Response threshold
                        ack= randi(2); %done=0;
                        if ack==1
                            if rand < A(i)/(R(2,1)+ A(i)) %&& done==0  %A(i) >= R(2,1)
                                actionY(i+1) = 1;
                                done=1;
                            elseif rand < B(i)/ (R(2,2)+B(i)) %&& done==0 % B(i) >= R(2,2)
                                actionY(i+1) = 2;
                                %done=1;
                            else
                                actionY(i+1) = 0;
                                %done=1;
                            end
                        elseif ack==2
                            if rand < B(i)/ (R(2,2)+B(i)) %&& done==0 % B(i) >= R(2,2)
                                actionY(i+1) = 2;
                                %done=1;
                            elseif rand < A(i)/(R(2,1)+ A(i)) %&& done==0  %A(i) >= R(2,1)
                                actionY(i+1) = 1;
                                %done=1;
                            else
                                actionY(i+1) = 0;
                                %done=1;
                            end
                        end
                    end
                    
                    if model==4
                        %%Model 4: Random selection - alternating
                        mid=randi(3);
                        %shift domain to [0,2]
                        mid = mid-1;
                        if mid>0
                            mid=lasttaskY;
                        end
                    end
                    
                    %if you are currently active in period i...
                    if model ~= 3 && actionY(i) > 0
                        if rand(1) < eta %the quitting prob.
                            actionY(i+1) = 0; %inactive in following period
                        else
                            actionY(i+1) = actionY(i); %else stay in current task
                        end
                        lasttaskY=actionY(i);
                    end
                    
                    %if you are currently inactive in period i...
                    if model ~= 3 && actionY(i) == 0
                        actionY(i+1) = mid; %select maximum action
                    end
                    
                    
                    if block_one_agent==1
                        actionY(i+1)=0;
                    end
                    
                    %optional - add delta - a spontenous rate of task demand
                    %increase
                    A(i+1) = A(i+1) + delta*n/m;
                    B(i+1) = B(i+1) + delta*n/m;
                    
                    %optional - track total energy consumption
                    %energy use is +0.5 when at rest and +1 when active
                    %             if actionX(i)>0
                    %                 energyX = energyX +1;
                    %             else
                    %                 energyX = energyX +0.5;
                    %             end
                    %
                    %             if actionY(i)>0
                    %                 energyY = energyY +1;
                    %             else
                    %                 energyY = energyY +0.5;
                    %             end
                    
                    %         %%%CALCULATE DOL index over time
                    %         ActivityMat = [countX1, countX2; countY1, countY2];
                    %         dol_metric = computeDOL(ActivityMat); %calls an external funciton
                    %         %D = [D dol_metric];
                    %         Dtime(i) = dol_metric;
                end
                cX1(iter) = countX1;
                cX2(iter) = countX2;
                cY1(iter) = countY1;
                cY2(iter) = countY2;
                cX0(iter) = X0(end);
                cY0(iter) = Y0(end);
                cA(iter) = A(end);
                cB(iter) = B(end);
                
                tFX(iter)=sum(X0)/T;
                tFY(iter)=sum(Y0)/T;
                tE(iter)=(tFX(iter)+tFY(iter))*0.5; %energy expenditure per-capita
                %tE(iter)=(cX0(iter)+cY0(iter))*0.5; %energy expenditure per-capita (end)
                tW(iter)=A(1)+B(1)-A(end)-B(end); %tokens of work completed at the end of simulation
                
            end
            
            %use median score
            countX1= median(cX1);
            countX2= median(cX2);
            countY1= median(cY1);
            countY2= median(cY2);
            countX0= median(cX0);
            countY0= median(cY0);
            countA= median(cA);
            countB= median(cB);
            totalFX=median(tFX);
            totalFY=median(tFY);
            totalE=median(tE);
            totalW=median(tW);
            
            
            %%%CALCULATE DIVISION OF LABOR METRIC%%%
            ActivityMat = [countX1, countX2; countY1, countY2];
            dol_metric = computeDOL(ActivityMat); %calls an external funciton
            D(ind1,ind2) = dol_metric;
            %%%CALCULATE GINI COEFFICIENT OF ENERGY EXPENDITURE%%%
            EnergySkew = [totalFX, totalFY];
            g_metric = computeGini(EnergySkew); %calls an external funciton
            G(ind1,ind2) = g_metric;
            F(ind1,ind2) =  totalE; %median per-capita energy expenditure
            W(ind1,ind2) =  totalW; %median task output
            
        end
    end
    
end

%%Code to visualize heatmap
zn = D;
zt = rot90(zn,-3);
yt=rot90(yvalues,2);
xt=xvalues;
figure()
h = heatmap(xt, yt,zt, 'Colormap', parula, 'ColorLimits',[0 1],...
    'ColorMethod','count', 'GridVisible','off', 'CellLabelColor','none');
ylabel('Practice efficiency ({\gamma})'), xlabel('Rest efficieny ({\beta})')
set(0,'DefaultAxesTitleFontWeight','normal');
set(gca,'FontSize',30)
title('Division of labor (DOL_i)')

zn = F;
zt = rot90(zn,-3)/max(max(zn));
yt=rot90(yvalues,2);
xt=xvalues;
figure()
h = heatmap(xt, yt,zt, 'Colormap', parula, 'ColorLimits',[0 1],...
    'ColorMethod','count', 'GridVisible','off', 'CellLabelColor','none');
ylabel('Practice efficiency ({\gamma})'), xlabel('Rest efficiency ({\beta})')
set(0,'DefaultAxesTitleFontWeight','normal');
set(gca,'FontSize',30)
title('Fatigue')

zn = W;
zt = rot90(zn,-3)/max(max(zn));
yt=rot90(yvalues,2);
xt=xvalues;
figure()
h = heatmap(xt, yt,zt, 'Colormap', parula, 'ColorLimits',[0 1],...
    'ColorMethod','count', 'GridVisible','off', 'CellLabelColor','none');
ylabel('Practice efficiency ({\gamma})'), xlabel('Rest efficiency ({\beta})')
set(0,'DefaultAxesTitleFontWeight','normal');
set(gca,'FontSize',30)
title('Task output')


zn = G;
zt = rot90(zn,-3)/max(max(zn));
yt=rot90(yvalues,2);
xt=xvalues;
figure()
h = heatmap(xt, yt,zt, 'Colormap', parula, 'ColorLimits',[0 1],...
    'ColorMethod','count', 'GridVisible','off', 'CellLabelColor','none');
ylabel('Practice efficiency ({\gamma})'), xlabel('Rest efficiency ({\beta})')
set(0,'DefaultAxesTitleFontWeight','normal');
set(gca,'FontSize',30)
title('Energy variance')
% 
zn = W./F;
zt = rot90(zn,-3); %rot90(zn,-3)/max(max(zt));
yt=rot90(yvalues,2);
xt=xvalues;
figure()
h = heatmap(xt, yt,zt, 'Colormap', parula, 'ColorLimits',[0,max(max(zt))],...
    'ColorMethod','count', 'GridVisible','off', 'CellLabelColor','none');
ylabel('Practice efficiency ({\gamma})'), xlabel('Rest efficiency ({\beta})')
set(0,'DefaultAxesTitleFontWeight','normal');
set(gca,'FontSize',30)
title('Work efficiency')