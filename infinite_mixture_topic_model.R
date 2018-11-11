#####���������g�s�b�N���f��#####
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
detach("package:gtools", unload=TRUE)
library(extraDistr)
library(monomvn)
library(glmnet)
library(reshape2)
library(dplyr)
library(plyr)
library(ggplot2)

####�f�[�^�̔���####
#set.seed(423943)
#�f�[�^�̐ݒ�
k <- 10   #�g�s�b�N��
d <- 2000   #������
v <- 250   #��b��
w <- rpois(d, 150)   #1����������̒P�ꐔ

#�p�����[�^�̐ݒ�
alpha0 <- round(runif(k, 0.1, 1.25), 3)   #�����̃f�B���N�����O���z�̃p�����[�^
alpha1 <- rep(0.25, v)   #�P��̃f�B���N�����O���z�̃p�����[�^

#�f�B���N�������̔���
theta0 <- theta <- rdirichlet(d, alpha0)   #�����̃g�s�b�N���z���f�B���N���������甭��
phi0 <- phi <- rdirichlet(k, alpha1)   #�P��̃g�s�b�N���z���f�B���N���������甭��

#�������z�̗�������f�[�^�𔭐�
WX <- matrix(0, nrow=d, ncol=v)
Z <- list()
for(i in 1:d){
  z <- t(rmultinom(w[i], 1, theta[i, ]))   #�����̃g�s�b�N���z�𔭐�
  zn <- z %*% c(1:k)   #0,1�𐔒l�ɒu��������
  zdn <- cbind(zn, z)   #apply�֐��Ŏg����悤�ɍs��ɂ��Ă���
  wn <- t(apply(zdn, 1, function(x) rmultinom(1, 1, phi[x[1], ])))   #�����̃g�s�b�N����P��𐶐�
  wdn <- colSums(wn)   #�P�ꂲ�Ƃɍ��v����1�s�ɂ܂Ƃ߂�
  WX[i, ] <- wdn  
  Z[[i]] <- zdn[, 1]
  print(i)
}

####EM�A���S���Y���Ńg�s�b�N���f���𐄒�####
####�g�s�b�N���f���̂��߂̃f�[�^�Ɗ֐��̏���####
##���ꂼ��̕������̒P��̏o�����x�N�g���ɕ��ׂ�
##�f�[�^����pID���쐬
ID_list <- list()
wd_list <- list()

#���l���Ƃɋ��lID����ђP��ID���쐬
for(i in 1:nrow(WX)){
  print(i)
  ID_list[[i]] <- rep(i, w[i])
  num1 <- (WX[i, ] > 0) * c(1:v) 
  num2 <- subset(num1, num1 > 0)
  W1 <- WX[i, (WX[i, ] > 0)]
  number <- rep(num2, W1)
  wd_list[[i]] <- number
}

#���X�g���x�N�g���ɕϊ�
ID_d <- unlist(ID_list)
wd <- unlist(wd_list)

##�C���f�b�N�X���쐬
doc_list <- list()
word_list <- list()
for(i in 1:length(unique(ID_d))) {doc_list[[i]] <- which(ID_d==i)}
for(i in 1:length(unique(wd))) {word_list[[i]] <- which(wd==i)}
gc(); gc()


####�}���R�t�A�������e�J�����@�Ŗ��������g�s�b�N���f���𐄒�####
##�P�ꂲ�Ƃɖޓx�ƕ��S�����v�Z����֐�
burden_fr <- function(theta, phi, wd, w, k){
  Bur <-  matrix(0, nrow=length(wd), ncol=k)   #���S�W���̊i�[�p
  for(kk in 1:k){
    #���S�W�����v�Z
    Bi <- rep(theta[, kk], w) * (phi[kk, c(wd)])   #�ޓx
    Bur[, kk] <- Bi   
  }
  Br <- Bur / rowSums(Bur)   #���S���̌v�Z
  bval <- list(Br=Br, Bur=Bur)
  return(bval)
}


##�A���S���Y���̐ݒ�
R <- 10000
keep <- 4
rbeta <- 1.5
iter <- 0
k0 <- 2   #�����g�s�b�N��

##���O���z�̐ݒ�
#�n�C�p�[�p�����[�^�̎��O���z
alpha01 <- rep(1.0, k0)
beta0 <- rep(0.5, v)
alpha01m <- matrix(alpha01, nrow=d, ncol=k0, byrow=T)
beta0m <- matrix(beta0, nrow=v, ncol=k0)

#�W���x�p�����[�^
tau1 <- 1
tau2 <- 2

##�p�����[�^�̏����l
theta.ini <- runif(k0, 0.5, 2)
phi.ini <- runif(v, 0.5, 1)
theta <- rdirichlet(d, theta.ini)   #�����g�s�b�N�̃p�����[�^�̏����l
phi <- rdirichlet(k0, phi.ini)   #�P��g�s�b�N�̃p�����[�^�̏����l

#�P�ꂲ�ƂɃg�s�b�N�̏o�������v�Z
word_rate <- burden_fr(theta, phi, wd, w, k0)$Br   #�����g�s�b�N�̏o����

#�������z����P��g�s�b�N���T���v�����O
vec <- 1/(1:k0)
word_cumsums <- rowCumsums(word_rate)
rand <- matrix(runif(nrow(word_rate)), nrow=nrow(word_rate), ncol=k0)   #��l����
Zi1 <- ((k0+1) - (word_cumsums > rand) %*% rep(1, k0)) %*% vec   #�g�s�b�N���T���v�����O
Zi1[Zi1!=1] <- 0

#�e�[�u�������X�V
Zi1[, 1]


##�������̍X�V
r0 <- c(colSums(Zi1), tau2)
pi0 <- extraDistr::rdirichlet(1, r0)

##�����g�s�b�N�̃p�����[�^���X�V
#�f�B�N�������z����theta���T���v�����O
for(i in 1:d){
  wsum0[i, ] <- colSums(Zi1[doc_list[[i]], ]) 
}
theta_prior <- alpha01m * matrix(pi0[, 1:k0], nrow=d, ncol=k0, byrow=T)
wsum <- cbind(wsum0 + theta_prior, tau1 * pi0[, k0+1])   #�f�B�N�������z�̃p�����[�^
theta_tau <- extraDistr::rdirichlet(d, wsum)[, k0+1]   #�f�B�N�������z����theta���T���v�����O
theta_tau


##�p�����[�^�̊i�[�p�z��
max_k <- 20
THETA <- array(0, dim=c(d, max_k, R/keep))
PHI <- array(0, dim=c(max_k, v, R/keep))
W_SEG <- matrix(0, nrow=sum(w), ncol=max_k)
storage.mode(W_SEG) <- "integer"
gc(); gc()

##MCMC����p�z��
wsum0 <- matrix(0, nrow=d, ncol=k0)
vf0 <- matrix(0, nrow=v, ncol=k0)
vec <- 1/1:k0

####�M�u�X�T���v�����O�Ńg�s�b�N���f���̃p�����[�^���T���v�����O####
for(rp in 1:R){
  #�g�s�b�N������X�V
  k0 <- ncol(theta) 
  k1 <- k0 + 1
  
  ##�P��g�s�b�N���T���v�����O
  #�P�ꂲ�ƂɃg�s�b�N�̏o�������v�Z
  word_rate_old <- burden_fr(theta, phi, wd, w, k0)$Bur   #�����g�s�b�N�̏o����
  word_rate_new <- pi0 * theta_tau[ID_d]   #�V�K�g�s�b�N�̏o����
  word_rate <- cbind(word_rate_old, word_rate_new)
  
  #�������z����P��g�s�b�N���T���v�����O
  word_cumsums <- rowCumsums(word_rate)
  rand <- matrix(runif(nrow(word_rate)), nrow=nrow(word_rate), ncol=k1)   #��l����
  Zi1 <- ((k1+1) - (word_cumsums > rand) %*% rep(1, k1)) %*% vec   #�g�s�b�N���T���v�����O
  Zi1[Zi1!=1] <- 0
  
  #Zi1 <- rmnom(nrow(word_rate), 1, word_rate)

  
  ##�����g�s�b�N�̃p�����[�^���X�V
  #�f�B�N�������z����theta���T���v�����O
  for(i in 1:d){
    wsum0[i, ] <- colSums(Zi1[doc_list[[i]], ]) 
  }
  wsum <- wsum0 + alpha01m   #�f�B�N�������z�̃p�����[�^
  theta <- extraDistr::rdirichlet(d, wsum)   #�f�B�N�������z����theta���T���v�����O
  
  ##�P��g�s�b�N�̃p�����[�^���X�V
  #�f�B�N�������z����phi���T���v�����O
  for(i in 1:v){
    vf0[i, ] <- colSums(Zi1[word_list[[i]], ])
  }
  vf <- t(vf0 + beta0m)   #�f�B�N�������z�̃p�����[�^
  phi <- extraDistr::rdirichlet(k, vf)   #�f�B�N�������z����phi���T���v�����O
  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^���i�[
  if(rp%%keep==0){
    mkeep1 <- rp/keep
    THETA[, , mkeep1] <- theta
    PHI[, , mkeep1] <- phi
    #W_SEG[mkeep2, ] <- word_z

    #�T���v�����O���ʂ��m�F
    print(rp)
    print(round(cbind(theta[1:10, ], theta0[1:10, ]), 3))
    #print(round(cbind(phi[, 1:10], phit[, 1:10]), 3))

  }
}


