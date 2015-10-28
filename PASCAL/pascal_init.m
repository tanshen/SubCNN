% initialize the PASCAL development kit 
tmp = pwd;
cd(opt.path_pascal);
addpath([cd '/VOCcode']);
VOCinit;
cd(tmp);
