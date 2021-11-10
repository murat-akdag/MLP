clc;
clear;
veriseti = xlsread('Cryotherapy.xlsx');
giris=veriseti(:,1:6);
cikis=veriseti(:,7);
giris=normalize(giris,'range');

giris_veri=giris(1:60,:);
test_veri=giris(61:90,:);

cikis_veri=cikis(1:60,1);
cikis_test=cikis(61:90,1);

%giris = [0,0;0,1;1,0;1,2];
%cikis = [0;1;1;0];

%% egitim
[w,b,katman_index]=MLP(giris_veri,cikis_veri);

%% test
test=test_veri(10,:);
for i=1:katman_index-1
    test=test*w{i}'+b{i};
    test=sigmoid(test);
end

disp(test);

