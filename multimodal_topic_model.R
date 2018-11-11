#####�}���`���[�_���g�s�b�N���f��#####
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
detach("package:gtools", unload=TRUE)
detach("package:bayesm", unload=TRUE)
library(extraDistr)
library(monomvn)
library(glmnet)
library(reshape2)
library(dplyr)
library(plyr)
library(ggplot2)

#set.seed(54876)

####�f�[�^�̔���####
#set.seed(423943)
#�f�[�^�̐ݒ�
m <- 3   #�����f�[�^��
k <- 8   #�g�s�b�N��
d <- 2500   #������

#�e�����f�[�^�̐ݒ�
#��b���̐ݒ�
v1 <- 250   
v2 <- 200
v3 <- 150

#1����������̒P�ꐔ
w1 <- rpois(d, 150)  
w2 <- rpois(d, 125)
w3 <- rpois(d, 100)

#�p�����[�^�̐ݒ�
alpha0 <- rep(0.5, k)   #�S�̂̃f�B���N�����O���z�̃p�����[�^
alpha1 <- rep(0.25, v1)   #����1�̒P��̃f�B���N�����O���z�̃p�����[�^
alpha2 <- rep(0.3, v2)   #����2�̒P��̃f�B�N�������O���z�̃p�����[�^
alpha3 <- rep(0.3, v3)   #����3�̒P��̃f�B�N�������O���z�̃p�����[�^

#�f�B���N�������̔���
thetat <- theta <- rdirichlet(d, alpha0)   #�����̃g�s�b�N���z���f�B���N���������甭��
phit1 <- phi1 <- rdirichlet(k, alpha1)   #�P��̃g�s�b�N���z���f�B���N���������甭��
phit2 <- phi2 <- rdirichlet(k, alpha2)   #�P��̃g�s�b�N���z���f�B���N���������甭��
phit3 <- phi3 <- rdirichlet(k, alpha3)   #�P��̃g�s�b�N���z���f�B���N���������甭��


#�������z�̗�������f�[�^�𔭐�
WX1 <- matrix(0, nrow=d, ncol=v1)
WX2 <- matrix(0, nrow=d, ncol=v2)
WX3 <- matrix(0, nrow=d, ncol=v3)
Z1 <- list()
Z2 <- list()
Z3 <- list()

for(i in 1:d){
  print(i)
  
  #����1�̃g�s�b�N���z�𔭐�
  z1 <- t(rmultinom(w1[i], 1, theta[i, ]))   #�����̃g�s�b�N���z�𔭐�
  
  #����1�̃g�s�b�N���z����P��𔭐�
  zn <- z1 %*% c(1:k)   #0,1�𐔒l�ɒu��������
  zdn <- cbind(zn, z1)   #apply�֐��Ŏg����悤�ɍs��ɂ��Ă���
  wn <- t(apply(zdn, 1, function(x) rmultinom(1, 1, phi1[x[1], ])))   #�����̃g�s�b�N����P��𐶐�
  wdn <- colSums(wn)   #�P�ꂲ�Ƃɍ��v����1�s�ɂ܂Ƃ߂�
  WX1[i, ] <- wdn  
  Z1[[i]] <- zdn[, 1]   #�����g�s�b�N���i�[
  
  
  #����2�̃g�s�b�N���z�𔭐�
  z2 <- t(rmultinom(w2[i], 1, theta[i, ]))   #�����̃g�s�b�N���z�𔭐�
  
  #����2�̃g�s�b�N���z����P��𔭐�
  zn <- z2 %*% c(1:k)   #0,1�𐔒l�ɒu��������
  zdn <- cbind(zn, z2)   #apply�֐��Ŏg����悤�ɍs��ɂ��Ă���
  wn <- t(apply(zdn, 1, function(x) rmultinom(1, 1, phi2[x[1], ])))   #�����̃g�s�b�N����P��𐶐�
  wdn <- colSums(wn)   #�P�ꂲ�Ƃɍ��v����1�s�ɂ܂Ƃ߂�
  WX2[i, ] <- wdn 
  Z2[[i]] <- zdn[, 1]   #�����g�s�b�N���i�[
  
  
  #����3�̃g�s�b�N���z�𔭐�
  z3 <- t(rmultinom(w3[i], 1, theta[i, ]))   #�����̃g�s�b�N���z�𔭐�
  
  #����3�̃g�s�b�N���z����P��𔭐�
  zn <- z3 %*% c(1:k)   #0,1�𐔒l�ɒu��������
  zdn <- cbind(zn, z3)   #apply�֐��Ŏg����悤�ɍs��ɂ��Ă���
  wn <- t(apply(zdn, 1, function(x) rmultinom(1, 1, phi3[x[1], ])))   #�����̃g�s�b�N����P��𐶐�
  wdn <- colSums(wn)   #�P�ꂲ�Ƃɍ��v����1�s�ɂ܂Ƃ߂�
  WX3[i, ] <- wdn 
  Z3[[i]] <- zdn[, 1]   #�����g�s�b�N���i�[
}

#�f�[�^�s��𐮐��^�s��ɕύX
storage.mode(WX1) <- "integer"
storage.mode(WX2) <- "integer"
storage.mode(WX3) <- "integer"

####�}���`���[�_��LDA����̂��߂̃f�[�^�Ɗ֐��̏���####
##���ꂼ��̕������̒P��̏o������ѕ⏕���̏o�����x�N�g���ɕ��ׂ�
##�f�[�^����pID���쐬
ID1_list <- list()
ID2_list <- list()
ID3_list <- list()
wd1_list <- list()
wd2_list <- list()
wd3_list <- list()

#���l���Ƃɋ��lID����ђP��ID���쐬
for(i in 1:d){
  print(i)
  
  #����1�̒P���ID�x�N�g�����쐬
  ID1_list[[i]] <- rep(i, w1[i])
  num1 <- (WX1[i, ] > 0) * (1:v1)
  num2 <- subset(num1, num1 > 0)
  W1 <- WX1[i, (WX1[i, ] > 0)]
  number <- rep(num2, W1)
  wd1_list[[i]] <- number
  
  #����2�̒P���ID�x�N�g�����쐬
  ID2_list[[i]] <- rep(i, w2[i])
  num1 <- (WX2[i, ] > 0) * (1:v2)
  num2 <- subset(num1, num1 > 0)
  W2 <- WX2[i, (WX2[i, ] > 0)]
  number <- rep(num2, W2)
  wd2_list[[i]] <- number
  
  #����3�̒P���ID�x�N�g�����쐬
  ID3_list[[i]] <- rep(i, w3[i])
  num1 <- (WX3[i, ] > 0) * (1:v3)
  num2 <- subset(num1, num1 > 0)
  W3 <- WX3[i, (WX3[i, ] > 0)]
  number <- rep(num2, W3)
  wd3_list[[i]] <- number
}

#���X�g���x�N�g���ɕϊ�
ID1_d <- unlist(ID1_list)
ID2_d <- unlist(ID2_list)
ID3_d <- unlist(ID3_list)
wd1 <- unlist(wd1_list)
wd2 <- unlist(wd2_list)
wd3 <- unlist(wd3_list)
storage.mode(ID1_d) <- "integer"
storage.mode(ID2_d) <- "integer"
storage.mode(ID3_d) <- "integer"
storage.mode(wd1) <- "integer"
storage.mode(wd2) <- "integer"
storage.mode(wd3) <- "integer"


##�C���f�b�N�X���쐬
doc1_list <- list()
doc2_list <- list()
doc3_list <- list()
word1_list <- list()
word2_list <- list()
word3_list <- list()
for(i in 1:length(unique(ID1_d))) {doc1_list[[i]] <- which(ID1_d==i)}
for(i in 1:length(unique(wd1))) {word1_list[[i]] <- which(wd1==i)}
for(i in 1:length(unique(ID2_d))) {doc2_list[[i]] <- which(ID2_d==i)}
for(i in 1:length(unique(wd2))) {word2_list[[i]] <- which(wd2==i)}
for(i in 1:length(unique(ID3_d))) {doc3_list[[i]] <- which(ID3_d==i)}
for(i in 1:length(unique(wd3))) {word3_list[[i]] <- which(wd3==i)}
gc(); gc()


####�}���R�t�A�������e�J�����@�Ń}���`���[�_��LDA�𐄒�####
##�P�ꂲ�Ƃɖޓx�ƕ��S�����v�Z����֐�
burden_fr <- function(theta, phi, wd, w, k){
  Bur <-  matrix(0, nrow=length(wd), ncol=k)   #���S�W���̊i�[�p
  for(kk in 1:k){
    #���S�W�����v�Z
    Bi <- rep(theta[, kk], w) * phi[kk, c(wd)]   #�ޓx
    Bur[, kk] <- Bi   
  }
  
  Br <- Bur / rowSums(Bur)   #���S���̌v�Z
  r <- colSums(Br) / sum(Br)   #�������̌v�Z
  bval <- list(Br=Br, Bur=Bur, r=r)
  return(bval)
}

##�A���S���Y���̐ݒ�
R <- 10000   #�T���v�����O��
keep <- 2
burnin <- 1000/keep
iter <- 0

##���O���z�̐ݒ�
#�n�C�p�[�p�����[�^�̎��O���z
alpha01 <- rep(1, k)
beta01 <- rep(0.5, v1)
beta02 <- rep(0.5, v2)
beta03 <- rep(0.5, v3)
alpha01m <- matrix(alpha01, nrow=d, ncol=k, byrow=T)
beta01m <- matrix(beta01, nrow=v1, ncol=k)
beta02m <- matrix(beta02, nrow=v2, ncol=k)
beta03m <- matrix(beta03, nrow=v3, ncol=k)

##�p�����[�^�̏����l��ݒ�
theta.ini <- runif(k, 0.5, 2)
phi1.ini <- runif(v1, 0.5, 1)
phi2.ini <- runif(v2, 0.5, 1)
phi3.ini <- runif(v3, 0.5, 1)
theta <- rdirichlet(d, theta.ini)   #�S�̂̕����g�s�b�N�̃p�����[�^�̏����l
phi1 <- rdirichlet(k, phi1.ini)   #����1�̒P��g�s�b�N�̃p�����[�^�̏����l
phi2 <- rdirichlet(k, phi2.ini)   #����2�̒P��g�s�b�N�̃p�����[�^�̏����l
phi3 <- rdirichlet(k, phi3.ini)   #����3�̒P��g�s�b�N�̃p�����[�^�̏����l


##�p�����[�^�̊i�[�p�z��
THETA <- array(0, dim=c(d, k, R/keep))
PHI1 <- array(0, dim=c(k, v1, R/keep))
PHI2 <- array(0, dim=c(k, v2, R/keep))
PHI3 <- array(0, dim=c(k, v3, R/keep))
W1_SEG <- matrix(0, nrow=sum(w1), ncol=k)
W2_SEG <- matrix(0, nrow=sum(w2), ncol=k)
W3_SEG <- matrix(0, nrow=sum(w3), ncol=k)
storage.mode(W1_SEG) <- "integer"
storage.mode(W2_SEG) <- "integer"
storage.mode(W3_SEG) <- "integer"
gc(); gc()

##MCMC����p�z��
vec <- 1/1:k
vf01 <- matrix(0, nrow=v1, ncol=k)
vf02 <- matrix(0, nrow=v2, ncol=k)
vf03 <- matrix(0, nrow=v3, ncol=k)
wsum1 <- wsum2 <- wsum3 <- matrix(0, nrow=d, ncol=k)


####�}���R�t�A�������e�J�����@�Ńp�����[�^���T���v�����O####
for(rp in 1:R){
  
  ##����1�̃g�s�b�N���T���v�����O
  #�P�ꂲ�ƂɃg�s�b�N�̏o�������v�Z
  word_rate1 <- burden_fr(theta, phi1, wd1, w1, k)$Br
  
  #�������z����P��g�s�b�N���T���v�����O
  word_cumsums <- rowCumsums(word_rate1)
  rand <- matrix(runif(nrow(word_rate1)), nrow=nrow(word_rate1), ncol=k)   #��l����
  Zi1 <- ((k+1) - (word_cumsums > rand) %*% rep(1, k)) %*% vec   #�g�s�b�N���T���v�����O
  Zi1[Zi1!=1] <- 0
  
  ##����1�̒P��g�s�b�N�̃p�����[�^���X�V
  #�f�B�N�������z����phi���T���v�����O
  for(i in 1:v1){
    vf01[i, ] <- colSums(Zi1[word1_list[[i]], ])
  }
  vf <- t(vf01 + beta01m)   #�f�B�N�������z�̃p�����[�^
  phi1 <- extraDistr::rdirichlet(k, vf)   #�f�B�N�������z����phi���T���v�����O
  
  
  ##����2�̃g�s�b�N���T���v�����O
  #�P�ꂲ�ƂɃg�s�b�N�̏o�������v�Z
  word_rate2 <- burden_fr(theta, phi2, wd2, w2, k)$Br
  
  #�������z����P��g�s�b�N���T���v�����O
  word_cumsums <- rowCumsums(word_rate2)
  rand <- matrix(runif(nrow(word_rate2)), nrow=nrow(word_rate2), ncol=k)   #��l����
  Zi2 <- ((k+1) - (word_cumsums > rand) %*% rep(1, k)) %*% vec   #�g�s�b�N���T���v�����O
  Zi2[Zi2!=1] <- 0
  
  ##����2�̒P��g�s�b�N�̃p�����[�^���X�V
  #�f�B�N�������z����phi���T���v�����O
  for(i in 1:v2){
    vf02[i, ] <- colSums(Zi2[word2_list[[i]], ])
  }
  vf <- t(vf02 + beta02m)   #�f�B�N�������z�̃p�����[�^
  phi2 <- extraDistr::rdirichlet(k, vf)   #�f�B�N�������z����phi���T���v�����O
  
  
  ##����3�̃g�s�b�N���T���v�����O
  #�P�ꂲ�ƂɃg�s�b�N�̏o�������v�Z
  word_rate3 <- burden_fr(theta, phi3, wd3, w3, k)$Br
  
  #�������z����P��g�s�b�N���T���v�����O
  word_cumsums <- rowCumsums(word_rate3)
  rand <- matrix(runif(nrow(word_rate3)), nrow=nrow(word_rate3), ncol=k)   #��l����
  Zi3 <- ((k+1) - (word_cumsums > rand) %*% rep(1, k)) %*% vec   #�g�s�b�N���T���v�����O
  Zi3[Zi3!=1] <- 0
  
  ##����3�̒P��g�s�b�N�̃p�����[�^���X�V
  #�f�B�N�������z����phi���T���v�����O
  for(i in 1:v3){
    vf03[i, ] <- colSums(Zi3[word3_list[[i]], ])
  }
  vf <- t(vf03 + beta03m)   #�f�B�N�������z�̃p�����[�^
  phi3 <- extraDistr::rdirichlet(k, vf)   #�f�B�N�������z����phi���T���v�����O
  
  
  ##�����������g�s�b�N���狤�ʂ�theta���T���v�����O
  #�f�B�N�������z�̃p�����[�^���v�Z
  for(i in 1:d){
    wsum1[i, ] <- colSums(Zi1[doc1_list[[i]], ])
    wsum2[i, ] <- colSums(Zi2[doc2_list[[i]], ])
    wsum3[i, ] <- colSums(Zi3[doc3_list[[i]], ])
  }
  wsum <- wsum1 + wsum2 + wsum3 + alpha01m   #�f�B�N�������z�̃p�����[�^
  theta <- extraDistr::rdirichlet(d, wsum)   #�f�B�N�������z����g�s�b�N�������T���v�����O
  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^���i�[
  if(rp%%keep==0){
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI1[, , mkeep] <- phi1
    PHI2[, , mkeep] <- phi2
    PHI3[, , mkeep] <- phi3
    
    if(rp >= burnin){
      W1_SEG <- W1_SEG + Zi1
      W2_SEG <- W2_SEG + Zi2
      W3_SEG <- W3_SEG + Zi3
    }
    
    #�T���v�����O���ʂ��m�F
    print(rp)
    print(round(cbind(theta[1:10, ], thetat[1:10, ]), 3))
    print(round(cbind(phi1[, 1:10], phit1[, 1:10]), 3))
  }
}

####�T���v�����O���ʂ̐���l�Ɨv��####
##�o�[���C�����Ԃ̐ݒ�
burnin <- 1000/keep
RS <- R/keep
z_range <- length(burnin:(R/keep))

##�T���v�����O���ʂ̉���
matplot(t(THETA[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^����l")
matplot(t(THETA[100, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^����l")
matplot(t(THETA[1000, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^����l")
matplot(t(THETA[2000, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^����l")
matplot(t(PHI1[1, 1:5, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^����l")
matplot(t(PHI2[2, 10:15, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^����l")
matplot(t(PHI3[3, 20:25, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^����l")

##�T���v�����O���ʂ̗v��
#�������̒P�ꂲ�Ƃ̃g�s�b�N�����m��
round(W1_SEG / rowSums(W1_SEG), 3)
round(W2_SEG / rowSums(W2_SEG), 3)
round(W3_SEG / rowSums(W3_SEG), 3)

#�����̃g�s�b�N�����m��
round(cbind(apply(THETA[, , burnin:RS], c(1, 2), mean), thetat), 3)

#�P��̃g�s�b�N�����m��
round(cbind(t(apply(PHI1[, , burnin:RS], c(1, 2), mean)), t(phit1)), 3)
round(cbind(t(apply(PHI2[, , burnin:RS], c(1, 2), mean)), t(phit2)), 3)
round(cbind(t(apply(PHI3[, , burnin:RS], c(1, 2), mean)), t(phit3)), 3)


