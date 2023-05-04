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
T = 100;
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
eta=0.25; %quitting prob. (try 0.5, 0.25)
%niter = 5000; %number of interations

model=1;  %1:self-maintenance, 2:random, 3:response-threshold, 4:task fidelity
block_one_agent=0; %for single-agent model, set equals to 1

TWmat = []; TEmat = [];

vals= 0.95;%[0.05 0.95];%linspace(0,1,5);
for val=1:length(vals)
    beta=vals(val);
    D=[]; %dol matrix to be populated
    Dtime = [];
    TWmat = []; TEmat = [];
    for k=1:niter
        
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
            
            %%%CALCULATE DOL index over time
            ActivityMat = [countX1, countX2; countY1, countY2];
            dol_metric = computeDOL(ActivityMat); %calls an external funciton
            %D = [D dol_metric];
            Dtime(i) = dol_metric;
            
        end
        totalFX=sum(X0);
        totalFY=sum(Y0);
        totalE=(totalFX+totalFY)*0.5/T; %energy expenditure per-capita
        totalW=A(1)+B(1)-A(end)-B(end); %tokens of work completed at the end of simulation
        TEmat = [TEmat; totalE];
        TWmat = [TWmat; totalW];
        D = [D;Dtime];
    end
    
    
    
    %visialize task stimulus over time
    %figure()
    % plot(tspan,A+B,'-k','LineWidth',4)
    % ylabel('Task stimulus: S_A+S_B ');
    % xlabel('Simulation time');
    % ylim([0,100])
    % set(gca,'box','off','FontSize',30)
    % hold on
    
    % %figure()
    % plot(tspan,X0-0.5,'-k','LineWidth',4)
    % hold on
    % plot(tspan,Y0,'-g','LineWidth',4)
    % ylabel('Task stimulus: S_A+S_B ');
    % xlabel('Simulation time');
    % %ylim([0,100])
    % set(gca,'box','off','FontSize',30)
    % hold on
    
    %visialize DoL over time
    % figure()
    % plot(tspan(1:end-1),Dtime,'-','Color',gold,'LineWidth',4)
    % hold on
    % ylabel('Division of labor ');
    % xlabel('Simulation time');
    % ylim([0,1])
    % set(gca,'box','off','FontSize',30)
    
    %visialize DoL over time for different parameters (test code)
    % figure()
    % ld = size(D);
    % for i=1:ld(1)
    %     h = plot(tspan(1:end-1),D(i,:)-0.01*i,'--o','Color', [0 0 0]+0.5*0.145*(i),'LineWidth',0.01);
    %     set(h, 'markerfacecolor', get(h, 'color'), 'MarkerSize',0.5); % Use same color
    %     hold on
    % end
    
    figure()
    hold on
    ld = size(D);
    cm = gray(ld(1));
    for i=1:ld(1)
        h = plot(tspan(1:end-1),D(i,:),'--o','Color', cm(i,:),'LineWidth',0.01);
        set(h, 'markerfacecolor', get(h, 'color'), 'MarkerSize',0.5); % Use same color
        hold on
    end
    plot(tspan(1:end-1),median(D(i,:),1),'-r','LineWidth',4)
    ylim([0,1])
    ylabel('Division of labor');
    xlabel('Simulation time');
    set(gca,'box','off','FontSize',30)

    %legend('1','2','3','4','5')
    
    %plot fitness outcomes by component (task production & energy use)
    Fitness = [mean(TWmat) mean(TEmat)];
    Fstd = [std(TWmat)*Fitness(1) std(TEmat)*Fitness(2)]*(1/sqrt(niter));
    % figure()
    % y=barwitherr(Fstd,Fitness,'FaceColor','flat');
    % %y = bar(Fitness','FaceColor','flat');
    % y.CData(1,:) = black;
    % y.CData(2,:) = grey;
    % ylabel('Fitness');
    % xlabel('Component'); %component labels {benefit, cost}
    % ylim([0,50])
    % set(gca,'box','off','FontSize',30)
    
    %plot 2D correlation
    % figure()
    % scatter(TEmat,TWmat)
    
    h=size(D);
    
    c=linspace(1,10,h(1));
    
    [Dsorted,I] = sort(D(:,end));
    Fsorted =TEmat(I);
    Wsorted=TWmat(I);
    
    cd=[Fsorted Wsorted Dsorted];
    %cd=sort(cd,1);
    
    figure()
    scatter(cd(:,1),cd(:,2),100,cd(:,3),'filled') %nice work :)
    colormap parula
    hold on
    set(gca,'box','off','FontSize',30)
    Test=[TEmat,TWmat];
    mdl = fitlm(TEmat,TWmat,'linear');
    %figure()
    %plot(mdl, 'color', black)
    plot(mdl)
    ylabel('Task output');
    xlabel('Fatigue');
    title('')
    set(gca,'box','off','FontSize',30)
    % ylim([0,max(TWmat)])
    % xlim([0,max(TEmat)])
    %legend('Simulation data','Fit','Confidence bounds')
  
%     dataHandle = findobj(h,'DisplayName','Data');
%     fitHandle = findobj(h,'DisplayName','Fit');
%     % The confidence bounds have 2 handles but only one of
%     % the handles contains the legend string.  The first
%     % line below finds that object and then searches for
%     % other objects in the plot that have the same linestyle
%     % and color.
%     cbHandles = findobj(h,'DisplayName','Confidence bounds');
%     cbHandles = findobj(h,'LineStyle',cbHandles.LineStyle, 'Color', cbHandles.Color);
%     
%     dataHandle.Color = 'k';
%     fitHandle.Color = orange; %orange
%     set(cbHandles, 'Color', 'b', 'LineWidth', 1)
%     set(gca,'box','off','FontSize',30)
 
    
    %boxplot for rest propensity - run 3 times to populate hbx array
    if val==1
        hb1=D(:,end);% phi=0
    end
    if val==2
        hb2=D(:,end);% phi=1
    end
    if val==3
        hb3=D(:,end); %phi=40
    end
    if val==4
        hb4=D(:,end); %phi=40
    end
    if val==5
        hb5=D(:,end); %phi=40
        %data=[hb1 hb2 hb3 hb4 hb5];
        data=[hb1 hb2];
        figure()
        %boxplot(data,[1 2 3],'Labels',{'No rest','Partial rest','Full rest'},'Whisker',3, 'Symbol','k+');
        boxplot(data,vals,'Whisker',3, 'Symbol','k+');
        ylabel('Division of labor');
        xlabel('Rest efficiency');
        set(gca,'FontSize',30)
    end
    
end
% %joint panel (two agent)

%recode action vector
twoind=find(actionX==2);
actionX(twoind)=-1;
twoind=find(actionY==2);
actionY(twoind)=-1;

figure()
subplot(4,4,[1 4])
plot(tspan,A+0.5,'r','LineWidth',1.125)
ylabel('Stimulus');
xlabel('time');
ylim([0,50])
hold on
plot(tspan,B-0.5,'b','LineWidth',1.125)
legend('Task 1','Task 2')
set(gca,'box','off','FontSize',25)
subplot(4,4,[5 6])
stem(tspan,actionX,'k','LineWidth',1.125)
ylabel('Action');
xlabel('time');
ylim([-1,1])
set(gca,'box','off','FontSize',25)
subplot(4,4,[9 10])
plot(tspan,X0,'k','LineWidth',1.125)
ylabel('Fatigue');
xlabel('time');
%ylim([0,50])
set(gca,'box','off','FontSize',25)
subplot(4,4,[13 14])
plot(tspan,1-S(X1),'r','LineWidth',1.125)
ylabel('Competence');
xlabel('time');
ylim([0,1])
hold on
plot(tspan,1-S(X2),'b','LineWidth',1.125)
legend('Task 1','Task 2')
set(gca,'box','off','FontSize',25)
subplot(4,4,[7 8])
stem(tspan,actionY,'k','LineWidth',1.125)
ylabel('Action');
xlabel('time');
ylim([-1,1])
set(gca,'box','off','FontSize',25)
subplot(4,4,[11 12])
plot(tspan,Y0,'k','LineWidth',1.125)
ylabel('Fatigue');
xlabel('time');
%ylim([0,50])
set(gca,'box','off','FontSize',25)
subplot(4,4,[15 16])
plot(tspan,1-S(Y1),'r','LineWidth',1.125)
ylabel('Competence');
xlabel('time');
ylim([0,1])
hold on
plot(tspan,1-S(Y2),'b','LineWidth',1.125)
legend('Task 1','Task 2')
set(gca,'box','off','FontSize',25)
hold off