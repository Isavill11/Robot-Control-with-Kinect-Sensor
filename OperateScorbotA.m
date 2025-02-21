ScorInit;
ScorSetGripper('Open');
start= [0,0,0,0,0]; %[base,shoulder,elbow,wrist pitch,wrist roll]
approach= [-0.1,0.5,0,-pi/2,0];
pick= [0,-0.1,-0.86,-2/pi,0];
move= [pi,-0.1,-0.86,-2/pi,0];

ScorSetBSEPR(start);
ScorWaitForMove;

ScorSetBSEPR(approach);
ScorWaitForMove;

ScorSetBSEPR(pick);
ScorWaitForMove;
ScorSetGripper('Close');
ScorWaitForMove;

ScorSetBSEPR(move);
ScorWaitForMove;

%ScorSetBSEPR(start);
%ScorWaitForMove;
ScorGoHome;
