% SARSA(Lambda) on Maze task (Tabular methods)
% Code by Sridhar
% 1 right,2 left,3 up,4 down.
% we will see the effect of varying lambda on this task with accumulating
% traces
alpha=0.1;
epsilon=0.1; % exploration policy
num_iters=35;
gamma=0.95;
allstates=zeros(6,9);
lambdachoices=[0,0.25,0.5,1];
steps_taken=zeros(num_iters,size(lambdachoices,2));


qvalues=zeros(54,4);
for k=1:size(lambdachoices,2)
lambda=lambdachoices(k);
qvalues=zeros(54,4);
    
    for hi=1:num_iters

    eltrace=zeros(54,4); %eligibility traces
    curstate=[3,1];      %start of the maze
    terminate=0; % will change to 1 on reaching a terminal state
    % First Choose S and A
    csi=sub2ind(size(allstates),curstate(1),curstate(2)); % start state index
    [~,curact]=max(qvalues(csi,:)); % the greedy action corresponding to start state

     if (rand(1)< epsilon)  % exploratory action.
            temp=randperm(4,4);
            curact=temp(1);   
     end

        num_step=0;
        while (terminate~=1)

            num_step=num_step+1; % number of steps takencu
            csi=sub2ind(size(allstates),curstate(1),curstate(2));
             [reward,next_state,signal]=transition(curstate,curact);% find the transition and the reward
             q_cur=qvalues(csi,curact);
             nsi=sub2ind(size(allstates),next_state(1),next_state(2)); 
            [q_next,nextact]= max(qvalues(nsi,:)); %Pick A' greedily

            if (rand(1)< epsilon) % pick  A' random action, epsilon = 0.1
            temp=randperm(4,4);
            nextact=temp(1);
            q_next=qvalues(nsi,nextact);
            end

            delta= reward + gamma*q_next - q_cur; % The TD Error


          % Update Q for all states using the eligibility trace;

          eltrace(csi,curact)=min(eltrace(csi,curact)+1,1);
           qvalues=qvalues+ alpha*delta*eltrace;
           eltrace=eltrace.*(gamma*lambda);

          curstate=next_state; % backup for S,S'
          curact=nextact; % backup for A,A'
          terminate=signal; % check if we have reached a terminal state S'
        end
       steps_taken(hi,k)=num_step;
       fprintf('Agent Henry has solved the maze in %d steps \n',num_step);
    end
     
end

plot(1:num_iters,smooth(steps_taken(1:end,1)),'b');
hold on;

plot(1:num_iters,steps_taken(1:end,2),'r');
plot(1:num_iters,steps_taken(1:end,3),'k');
plot(1:num_iters,steps_taken(1:end,4),'c');
xlabel('Number of episodes elapsed'),ylabel('Steps per episode'),title('Agent with Replacing traces on Maze task');
legend('lambda=0','lambda=0.25','lambda=0.75','lambda=1');
hold off;



