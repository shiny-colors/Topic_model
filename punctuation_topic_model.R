#####Punctuation�g�s�b�N���f��#####
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
library(Matrix)
library(bayesm)
library(extraDistr)
library(reshape2)
library(dplyr)
library(plyr)
library(ggplot2)

#set.seed(8079)

####�f�[�^�̔���####
#�����f�[�^�̐ݒ�
k <- 10   #�g�s�b�N��
d <- 1000   #������
a <- rpois(d, rgamma(d, 22, 2.5))   #���͐�
a[a < 5] <- ceiling(runif(sum(a < 3), 5, 15))
s <- sum(a)   #�����͐�
v <- 300   #��b��
w <- rpois(s, 21)   #1���͂�����̒P�ꐔ
w[w==0] <- 1
f <- sum(w)   #���P�ꐔ

##ID�̐ݒ�
id_a <- rep(1:s, w)
id_u <- rep(1:d, a)
id_d <- c()
id_w <- c()
id_s <- c()

for(i in 1:d){
  freq <- w[rep(1:d, a)==i]
  id_d <- c(id_d, rep(i, sum(freq)))
  id_w <- c(id_w, 1:sum(freq))
  id_s <- c(id_s, 1:a[i])
}

##�p�����[�^�̐ݒ�
#�f�B���N�����O���z�̐ݒ�
alpha1 <- rep(0.2, k)   #�����g�s�b�N�̃f�B���N�����O���z�̃p�����[�^
alpha2 <- rep(0.3, v)   #�P��̃f�B���N�����O���z�̃p�����[�^

#�f�B���N�������̔���
thetat <- theta <- extraDistr::rdirichlet(d, alpha1)   #�������Ƃ̃g�s�b�N���z
phit <- phi <- extraDistr::rdirichlet(k, alpha2)   #�P�ꂲ�Ƃ̃g�s�b�N���z

##�������z����g�s�b�N����ђP��f�[�^�𔭐�
WX <- matrix(0, nrow=d, ncol=v)
AX <- matrix(0, nrow=s, ncol=v)
Z0 <- list()

for(i in 1:d){
  print(i)

  #�����g�s�b�N���z�𔭐�
  z <- rmnom(a[i], 1, theta[i, ])
  zd <- as.numeric(z %*% 1:k)
  
  #�����̃g�s�b�N����P��𐶐�
  an <- rmnom(sum(id_u==i), w[id_u==i], phi[zd, ])
  AX[id_u==i, ] <- an
  WX[i, ] <- colSums(an)
  Z0[[i]] <- z
}

#�f�[�^�s���ϊ�
Z <- do.call(rbind, Z0)
storage.mode(WX) <- "integer"
storage.mode(AX) <- "integer"


####�g�s�b�N���f������̂��߂̃f�[�^�Ɗ֐��̏���####
##�f�[�^����pID�̍쐬
ID1_list <- list()
ID2_list <- list()
wd_list <- list()

#�������Ƃɕ���ID����ђP��ID���쐬
for(i in 1:nrow(AX)){
  print(i)
  
  #�P���ID�x�N�g�����쐬
  ID1_list[[i]] <- rep(id_u[i], w[i])
  ID2_list[[i]] <- rep(id_s[i], w[i])

  num1 <- (AX[i, ] > 0) * (1:v)
  num2 <- which(num1 > 0)
  A1 <- AX[i, (AX[i, ] > 0)]
  number <- rep(num2, A1)
  wd_list[[i]] <- number
}

#���X�g���x�N�g���ɕϊ�
ID1_d <- unlist(ID1_list)
ID2_d <- unlist(ID2_list)
wd <- unlist(wd_list)

##�C���f�b�N�X���쐬
doc1_list <- list()
doc2_list <- list()
word_list <- list()
for(i in 1:length(unique(ID1_d))) {doc1_list[[i]] <- which(ID1_d==i)}
for(i in 1:length(unique(id_a))) {doc2_list[[i]] <- which(id_a==i)}
for(i in 1:length(unique(wd))) {word_list[[i]] <- which(wd==i)}
gc(); gc()


####�}���R�t�A�������e�J�����@��Punctuation�g�s�b�N���f���𐄒�####
##���͂��Ƃɖޓx�ƕ��S�����v�Z����֐�
burden_fr <- function(theta, phi, wd, w, k){
  Bur <-  matrix(0, nrow=length(wd), ncol=k)   #���S�W���̊i�[�p
  for(j in 1:k){
    #���S�W�����v�Z
    Bi <- rep(theta[, j], w) * phi[j, wd]   #�ޓx
    Bur[, j] <- Bi   
  }
  
  Br <- Bur / rowSums(Bur)   #���S���̌v�Z
  r <- colSums(Br) / sum(Br)   #�������̌v�Z
  bval <- list(Br=Br, Bur=Bur, r=r)
  return(bval)
}

##�A���S���Y���̐ݒ�
R <- 10000   #�T���v�����O��
keep <- 2   #2���1��̊����ŃT���v�����O���ʂ��i�[
iter <- 0
disp <- 10
burnin <- 1000/keep

##���O���z�̐ݒ�
#�n�C�p�[�p�����[�^�̎��O���z
alpha01m <- matrix(1.0, nrow=d, ncol=k, byrow=T)
beta0m <- matrix(0.5, nrow=k, ncol=v)

##�p�����[�^�̏����l�̐ݒ�
theta <- rdirichlet(d, rep(10, k))   #�����g�s�b�N�̃p�����[�^�̏����l
phi <- rdirichlet(k, colSums(WX)/100)   #�P��̏o�����̃p�����[�^�̏����l

##�p�����[�^�̊i�[�p�z��
THETA <- array(0, dim=c(d, k, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
SEG <- matrix(0, nrow=s, ncol=k)
storage.mode(SEG) <- "integer"
gc(); gc()

##MCMC����p�̔z��
wf0 <- matrix(0, nrow=d, ncol=k)
vf0 <- matrix(0, nrow=k, ncol=v)


####�M�u�X�T���v�����O�Ńp�����[�^���T���v�����O####
for(rp in 1:R){

  ##���͂��ƂɃg�s�b�N���T���v�����O
  #���̓��x���ł̖ޓx�ƕ��S�����v�Z
  LLi <- AX %*% t(log(phi))   #���͂��Ƃ̑ΐ��ޓx
  LLi_max <- apply(LLi, 1, max)   
  Bur <- theta[id_u, ] * exp(LLi - LLi_max)   #���̓��x���̖ޓx
  Br <- Bur / rowSums(Bur)   #���S��
  
  #�������z���當�̓g�s�b�N���T���v�����O
  Zi <- rmnom(s, 1, Br)

  #�g�s�b�N���z�̃p�����[�^���T���v�����O
  Zi_word <- Zi[id_a, ]
  for(i in 1:d){
    wf0[i, ] <- colSums(Zi_word[doc1_list[[i]], ])
  }
  wf <- wf0 + alpha01m
  theta <- extraDistr::rdirichlet(d, wf)
  
  
  ##�g�s�b�N���ƂɒP��̏o�������T���v�����O
  for(j in 1:v){
    vf0[, j] <- colSums(Zi_word[word_list[[j]], ])
  }
  vf <- vf0 + beta0m
  phi <- extraDistr::rdirichlet(k, vf)
  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^���i�[
  if(rp%%keep==0){
    #�T���v�����O���ʂ̊i�[
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    
    #�g�s�b�N�����̓o�[���C�����Ԃ𒴂�����i�[����
    if(rp >= burnin){
      SEG <- SEG + Zi
    }
    
    #�T���v�����O���ʂ��m�F
    if(rp%%disp==0){
      print(rp)
      print(round(cbind(theta[1:10, ], thetat[1:10, ]), 3))
      print(round(cbind(phi[, 1:10], phit[, 1:10]), 3))
    }
  }
}

####�T���v�����O���ʂ̉����Ɨv��####
burnin <- 1000/keep
RS <- R/keep

##�T���v�����O���ʂ̉���
matplot(t(THETA[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA[100, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA[500, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA[1000, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI[, 1, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI[, 100, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI[, 300, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")

##�T���v�����O���ʂ̗v�񓝌v��
round(cbind(apply(THETA[, , burnin:RS], c(1, 2), mean), thetat), 3)
round(cbind(t(apply(PHI[, , burnin:RS], c(1, 2), mean)), t(phi)), 3)
cbind(round(SEG / sum(SEG[1, ]), 3), Z)
