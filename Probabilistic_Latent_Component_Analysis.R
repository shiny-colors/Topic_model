#####�m���I���ݗv�f���(PLCA���f��)#####
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
library(Matrix)
library(bayesm)
library(HMM)
library(extraDistr)
library(reshape2)
library(dplyr)
library(plyr)
library(ggplot2)

#set.seed(86751)

####�f�[�^�̔���####
k <- 10   #�g�s�b�N��
d <- 2000   #������
v <- 500   #��b��
f <- 300000   #�v�f��

##�p�����[�^�̐ݒ�
#�f�B���N�����z�̃p�����[�^
alpha01 <- rep(3.0, k)
alpha11 <- rep(0.25, k)
alpha12 <- rep(0.15, v)

#�p�����[�^�𐶐�
pi <- pit <- extraDistr::rdirichlet(1, alpha01)
theta0 <- extraDistr::rdirichlet(d, alpha11) * matrix(runif(d, 0.5, 2.5), nrow=d, ncol=k)
theta <- thetat <- t(theta0 / matrix(colSums(theta0), nrow=d, ncol=k, byrow=T))
phi <- phit <- extraDistr::rdirichlet(k, alpha12)
Mu <- array(0, dim=c(d, v, k))
Par <- matrix(0, nrow=k, ncol=d*v)


##���f���Ɋ�Â��f�[�^�𐶐�
#�v�f���Ƃɑ������z����g�s�b�N�𐶐�
Z <- rmnom(f, 1, pi)
z <- as.numeric(Z %*% 1:k) 

#�g�s�b�N�Ɋ�Â��P��𐶐�
ID_d <- rep(0, f)
wd <- rep(0, f)

for(j in 1:k){
  print(j)
  index <- which(z==j)
  #�����̊����𐶐�
  Z1 <- rmnom(length(index), 1, theta[j, ])
  ID_d[index] <- as.numeric(Z1 %*% 1:d)
  
  #�P��̊����𐶐�
  Z2 <- rmnom(length(index), 1, phi[j, ])
  wd[index] <- as.numeric(Z2 %*% 1:v)
}

#�����s����쐬
WX <- matrix(0, nrow=d, ncol=v)
for(i in 1:f){
  WX[ID_d[i], wd[i]] <- WX[ID_d[i], wd[i]] + 1
}
hist(rowSums(WX), xlab="�P�ꐔ", main="�P��p�x�̕��z", col="grey", breaks=20)

##�C���f�b�N�X���쐬
doc_list <- list()
word_list <- list()
wd_list <- list()
for(i in 1:d){doc_list[[i]] <- which(ID_d==i)}
for(i in 1:v){word_list[[i]] <- which(wd==i)}
for(i in 1:v){wd_list[[i]] <- ID_d[word_list[[i]]]}


####�}���R�t�A�������e�J�����@��PLCA�𐄒�####
##�A���S���Y���̐ݒ�
R <- 10000
keep <- 2  
iter <- 0
burnin <- 1000/keep
disp <- 10

##�p�����[�^�̐^�l
theta <- t(thetat)
phi <- phit
pi <- pit

##�����l��ݒ�
theta0 <- t(extraDistr::rdirichlet(k, rowSums(WX)/sum(WX) * 200)) + 0.00001
theta <- theta0 / matrix(colSums(theta0), nrow=d, ncol=k)
phi0 <- extraDistr::rdirichlet(k, colSums(WX)/sum(WX) * 100) + 0.00001
phi <- phi0 / matrix(rowSums(phi0), nrow=k, ncol=v)
pi <- rep(1/k, k)

##���O���z�̐ݒ�
#�n�C�p�[�p�����[�^�̎��O���z
alpha01 <- 0.1 
beta01 <- 0.1
beta02 <- 1


##�p�����[�^�̊i�[�p�z��
THETA <- array(0, dim=c(d, k, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
PI <- matrix(0, nrow=R/keep, ncol=k)
SEG <- matrix(0, nrow=f, ncol=k)
storage.mode(SEG) <- "integer"


#�ΐ��ޓx�̊�l
par1 <- rowSums(WX)/sum(WX)
par2 <- colSums(WX)/sum(WX)
LLst0 <- rep(0, f)
for(i in 1:v){
  n <- length(wd_list[[i]])
  LLst0[word_list[[i]]] <- par1[wd_list[[i]]] * par2[i]   #�ޓx
}
LLst <- sum(log(LLst0))


####�M�u�X�T���v�����O��HTM���f���̃p�����[�^���T���v�����O####
for(rp in 1:R){
  
  ##�d�ݕt���ޓx����g�s�b�N���z�𐶐�
  #PLCA�̏d�ݕt���ޓx�ƃg�s�b�N�����m�����v�Z
  Li <- matrix(0, nrow=f, ncol=k)
  for(i in 1:v){
    n <- length(wd_list[[i]])
    r <- matrix(pi, nrow=n, ncol=k, byrow=T)
    Li[word_list[[i]], ] <- r * theta[wd_list[[i]], ] * matrix(phi[, i], nrow=n, ncol=k, byrow=T)   #�d�ݕt���ޓx
  }
  topic_rate <- Li / rowSums(Li)   #�g�s�b�N�����m��

  #�������z����g�s�b�N�𐶐�
  Zi <- rmnom(f, 1, topic_rate)
  z_vec <- as.numeric(Zi %*% 1:k)

  
  ##�p�����[�^���T���v�����O
  #������pi���T���v�����O
  psum <- colSums(Zi) + beta02
  pi <- extraDistr::rdirichlet(1, psum)
  
  #�������ztheta���T���v�����O
  wsum0 <- matrix(0, nrow=d, ncol=k)
  for(i in 1:d){
    wsum0[i, ] <- colSums(Zi[doc_list[[i]], ])
  }
  wsum <- t(wsum0)+ beta01
  theta <- t(extraDistr::rdirichlet(k, wsum))
  
  #�P�ꕪ�zphi���T���v�����O
  vf0 <- matrix(0, nrow=k, ncol=v)
  for(j in 1:v){
    vf0[, j] <- colSums(Zi[word_list[[j]], , drop=FALSE])
  }
  vf <- vf0 + alpha11
  phi <- extraDistr::rdirichlet(k, vf)
  
  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^���i�[
  if(rp%%keep==0){
    #�T���v�����O���ʂ̊i�[
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    PI[mkeep, ] <- pi
    
    #�g�s�b�N�����̓o�[���C�����Ԃ𒴂�����i�[����
    if(mkeep >= burnin & rp%%keep==0){
      SEG <- SEG + Zi
    }
    
    #�T���v�����O���ʂ��m�F
    if(rp%%disp==0){
      print(rp)
      print(c(sum(log(rowSums(Li))), LLst))
      print(round(cbind(theta[1:10, ], t(thetat[, 1:10])), 3))
      print(round(cbind(phi[, 1:10], phit[, 1:10]), 3))
      round(print(rbind(pi, pit)), 3)
    }
  }
}




