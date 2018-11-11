#####Four Levels Pachinko Allocation Model#####
options(warn=0)
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

#set.seed(93441)
####�f�[�^�̔���####
##�f�[�^�̐ݒ�
L <- 2
k1 <- 4   #��ʃg�s�b�N��
k2 <- 15   #���ʃg�s�b�N��
d <- 5000   #������
v <- 1200   #��b��
w <- rpois(d, rgamma(d, 50, 0.3))   #����������̒P�ꐔ
f <- sum(w)   #���P�ꐔ
vec_k1 <- rep(1, k1)
vec_k2 <- rep(1, k2)

#����ID�̐ݒ�
d_id <- rep(1:d, w)
a_id <- as.numeric(unlist(tapply(1:f, d_id, rank)))

##�p�����[�^�̐ݒ�
#�f�B���N�����z�̃p�����[�^��ݒ�
alpha1 <- rep(0.2, k1)
alpha2 <- rep(0.15, k2)
beta1 <- rep(0.2, k2)
beta2 <- rep(0.05, v)

##���f���Ɋ�Â��f�[�^�𐶐�
rp <- 0
repeat { 
  rp <- rp + 1
  print(rp)
  
  #�f�B���N�����z����p�����[�^�𐶐�
  theta1 <- thetat1 <- extraDistr::rdirichlet(d, alpha1)
  theta2 <- thetat2 <- array(0, dim=c(d, k2, k1))
  for(j in 1:k1){
    theta2[, , j] <- thetat2[, , j] <- extraDistr::rdirichlet(d, alpha2)
  }
  gamma <- gammat <- extraDistr::rdirichlet(k1, beta1)
  phi <- extraDistr::rdirichlet(k2, beta2)
  
  #�P��o���m�����Ⴂ�g�s�b�N�����ւ���
  index <- which(colMaxs(phi) < (k2*10)/f)
  for(j in 1:length(index)){
    phi[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, rep(2.0, k2))) %*% 1:k2), index[j]] <- (k2*10)/f
  }
  phit <- phi
  
  ##�������ƂɃg�s�b�N�ƒP��𐶐�
  Z1_list <- list()
  Z2_list <- list()
  word_list <- list()
  WX <- matrix(0, nrow=d, ncol=v)
  
  for(i in 1:d){
    #��ʃg�s�b�N�𐶐�
    z1 <- rmnom(w[i], 1, theta1[i, ])
    z1_vec <- as.numeric(z1 %*% 1:k1)
    
    #���ʃg�s�b�N�𐶐�
    z2 <- rmnom(w[i], 1, t(theta2[i, , z1_vec]))
    z2_vec <- as.numeric(z2 %*% 1:k2)
    
    #�P��𐶐�
    word <- rmnom(w[i], 1, phi[z2_vec, ])
    
    #�f�[�^���i�[
    Z1_list[[i]] <- z1
    Z2_list[[i]] <- z2
    word_list[[i]] <- as.numeric(word %*% 1:v)
    WX[i, ] <- colSums(word)
  }
  if(min(colSums(WX)) > 0){
    break
  }
}

#�f�[�^��ϊ�
Z1 <- do.call(rbind, Z1_list)
Z2 <- do.call(rbind, Z2_list)
storage.mode(WX) <- "integer"
wd <- unlist(word_list)
word_data <- sparseMatrix(1:f, wd, x=rep(1, f), dims=c(f, v))
word_data_T <- t(word_data)
rm(word_list); rm(Z1_list); rm(Z2_list)
gc(); gc()


####�}���R�t�A�������e�J�����@��PAM�𐄒�####
##�P�ꂲ�Ƃɖޓx�ƕ��S�����v�Z����֐�
burden_fr <- function(theta, phi, wd, w, k, vec_k){
  #���S�W�����v�Z
  Bur <- theta[w, ] * t(phi)[wd, ]   #�ޓx
  Br <- Bur / as.numeric(Bur %*% vec_k)   #���S��
  bval <- list(Br=Br, Bur=Bur)
  return(bval)
}

##�A���S���Y���̐ݒ�
R <- 5000
keep <- 2  
iter <- 0
burnin <- 1000/keep
disp <- 10

##�C���f�b�N�X�̐ݒ�
d_data <- sparseMatrix(1:f, d_id, x=rep(1, f), dims=c(f, d))
d_data_T <- t(d_data)

##���O���z�̐ݒ�
alpha1 <- 0.5
alpha2 <- 0.25
beta1 <- 0.1

##�p�����[�^�̐^�l
theta1 <- thetat1
theta2 <- thetat2
phi <- phit
Zi1 <- Z1 
Zi2 <- Z2

##�����l�̐ݒ�
theta1 <- extraDistr::rdirichlet(d, rep(2.0, k1))
theta2 <- array(0, dim=c(d, k2, k1))
for(j in 1:k1){
  theta2[, , j] <- extraDistr::rdirichlet(d, rep(2.0, k2))
}
phi <- extraDistr::rdirichlet(k2, rep(2.0, v))
Zi1 <- rmnom(f, 1, theta1[d_id, ])
z1_vec <- as.numeric(Z1 %*% 1:k1)

##�p�����[�^�̊i�[�p�z��
THETA1 <- array(0, dim=c(d, k1, R/keep))
THETA2 <- array(0, dim=c(d, k2, k1, R/keep))
PHI <- array(0, dim=c(k2, v, R/keep))
SEG1 <- matrix(0, nrow=f, ncol=k1)
SEG2 <- matrix(0, nrow=f, ncol=k2)

##�ΐ��ޓx�̊�l
#���j�O�������f���̑ΐ��ޓx
LLst <- sum(word_data %*% log(colSums(word_data) / f))

#�x�X�g���f���̑ΐ��ޓx
theta_topic <- thetat2[d_id, , ]
theta_k2 <- matrix(0, nrow=f, ncol=k2)
for(j in 1:k1){
  theta_k2 <- theta_k2 + theta_topic[, , j] * Z1[, j]
}
LLbest <- sum(log(as.numeric((theta_k2 * t(phit)[wd, ]) %*% vec_k2)))   #�ΐ��ޓx


####�M�u�X�T���v�����O�Ńp�����[�^���T���v�����O####
for(rp in 1:R){

  ##���ʃg�s�b�N���T���v�����O
  #��ʃg�s�b�N�����Ɋ�Â����ʃg�s�b�N���z��ݒ�
  theta_topic <- theta2[d_id, , ]
  theta_k2 <- matrix(0, nrow=f, ncol=k2)
  for(j in 1:k1){
    theta_k2 <- theta_k2 + theta_topic[, , j] * Zi1[, j]
  }
  
  #�g�s�b�N���z�̖ޓx�ƕ��S���𐄒�
  Lho2 <- theta_k2 * t(phi)[wd, ]   #�g�s�b�N�������Ƃ̖ޓx
  topic_rate <- Lho2 / as.numeric(Lho2 %*% vec_k2)   #���S��
  
  #�������z���g�s�b�N���T���v�����O
  Zi2 <- rmnom(f, 1, topic_rate)
  z2_vec <- as.numeric(Zi2 %*% 1:k2)
  
  
  ##��ʃg�s�b�N���T���v�����O
  #���ʃg�s�b�N�����Ɋ�Â���ʂ̃g�s�b�N�o�����z��ݒ�
  theta_k1 <- matrix(0, nrow=f, ncol=k1)
  for(j in 1:k2){
    theta_k1 <- theta_k1 + theta_topic[, j, ] * Zi2[, j] 
  }
  
  #�g�s�b�N���z�̖ޓx�ƕ��S���𐄒�
  Lho1 <- theta1[d_id, ] * theta_k1   #�g�s�b�N�������Ƃ̖ޓx
  topic_rate <- Lho1 / as.numeric(Lho1 %*% vec_k1)   #���S��
  
  #�������z���g�s�b�N���T���v�����O
  Zi1 <- rmnom(f, 1, topic_rate)
  z1_vec <- as.numeric(Zi1 %*% 1:k1)
  
  
  ##�p�����[�^���T���v�����O
  #��ʃg�s�b�N�̃p�����[�^���T���v�����O
  wsum1 <- as.matrix(d_data_T %*% Zi1) + alpha1   #�f�B���N�����z�̃p�����[�^
  theta1 <- extraDistr::rdirichlet(d, wsum1)   #�g�s�b�N���z���T���v�����O
  
  #��ʃg�s�b�N���Ƃɉ��ʃg�s�b�N�̃p�����[�^���T���v�����O
  for(j in 1:k1){
    wsum2 <- as.matrix(d_data_T %*% (Zi1[, j] * Zi2)) + alpha2     #�f�B�N�������z�̃p�����[�^
    theta2[, , j] <- extraDistr::rdirichlet(d, wsum2)   #�g�s�b�N���z���T���v�����O
  }
  
  ##�P�ꕪ�z�̃p�����[�^���T���v�����O
  #�f�B�N�������z�̃p�����[�^
  vsum <- as.matrix(t(word_data_T %*% Zi2)) + beta1
  phi <- extraDistr::rdirichlet(k2, vsum)   #�f�B�N�������z����P�ꕪ�z���T���v�����O
  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^���i�[
  if(rp%%keep==0){
    #�T���v�����O���ʂ̊i�[
    mkeep <- rp/keep
    THETA1[, , mkeep] <- theta1
    THETA2[, , , mkeep] <- theta2
    PHI[, , mkeep] <- phi
    
    #�g�s�b�N�����̓o�[���C�����Ԃ𒴂�����i�[����
    if(rp%%keep==0 & rp >= burnin){
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
    }
  }
  
  if(rp%%disp==0){
    #�ΐ��ޓx���v�Z
    LL <- sum(log(as.numeric((theta_k2 * t(phi)[wd, ]) %*% vec_k2)))
    
    #�T���v�����O���ʂ��m�F
    print(rp)
    print(c(LL, LLbest, LLst))
    print(round(cbind(theta1[1:5, ], thetat1[1:5, ]), 3))
    print(round(cbind(phi[, 1:10], phit[, 1:10]), 3))
  }
}

####�T���v�����O���ʂ̉����Ɨv��####
burnin <- 1000/keep
RS <- R/keep

##�T���v�����O���ʂ̉���
#��ʃg�s�b�N�̃g�s�b�N���z�̃T���v�����O����
matplot(t(THETA1[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^", main="��ʃg�s�b�N�̃T���v�����O����")
matplot(t(THETA1[5, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^", main="��ʃg�s�b�N�̃T���v�����O����")
matplot(t(THETA1[10, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^", main="��ʃg�s�b�N�̃T���v�����O����")
matplot(t(THETA1[15, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^", main="��ʃg�s�b�N�̃T���v�����O����")
matplot(t(THETA1[20, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^", main="��ʃg�s�b�N�̃T���v�����O����")

#���ʃg�s�b�N�̃g�s�b�N���z�̃T���v�����O����
matplot(t(THETA2[1, , 1, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^", main="���ʃg�s�b�N�̃T���v�����O����")
matplot(t(THETA2[1, , 2, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^", main="���ʃg�s�b�N�̃T���v�����O����")
matplot(t(THETA2[1, , 3, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^", main="���ʃg�s�b�N�̃T���v�����O����")
matplot(t(THETA2[1, , 4, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^", main="���ʃg�s�b�N�̃T���v�����O����")
matplot(t(THETA2[10, , 1, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^", main="���ʃg�s�b�N�̃T���v�����O����")
matplot(t(THETA2[10, , 2, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^", main="���ʃg�s�b�N�̃T���v�����O����")
matplot(t(THETA2[10, , 3, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^", main="���ʃg�s�b�N�̃T���v�����O����")
matplot(t(THETA2[10, , 4, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^", main="���ʃg�s�b�N�̃T���v�����O����")

#�P�ꕪ�z�̃T���v�����O����
matplot(t(PHI[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^", main="�P�ꕪ�z�̃T���v�����O����")
matplot(t(PHI[5, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^", main="�P�ꕪ�z�̃T���v�����O����")
matplot(t(PHI[10, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^", main="�P�ꕪ�z�̃T���v�����O����")
matplot(t(PHI[15, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^", main="�P�ꕪ�z�̃T���v�����O����")

##���㕪�z�̗v�񓝌v��
#�g�s�b�N���z�̎��㕽��
topic_mu1 <- apply(THETA1[, , burnin:RS], c(1, 2), mean)
topic_mu2 <- array(0, dim=c(d, k2, k1))
for(j in 1:k1){
  topic_mu2[, , j] <- apply(THETA2[, , j, burnin:RS], c(1, 2), mean)
}
round(topic_mu1, 3)
round(topic_mu2[, , 1], 3)

#�P�ꕪ�z�̎��㕽��
round(phi_mu <- t(apply(PHI[, , burnin:RS], c(1, 2), mean)), 3)
round(cbind(phi_mu, t(phit)), 3)

#�g�s�b�N�����̎��㕪�z
topic_rate1 <- SEG1 / rowSums(SEG1) 
topic_allocation1 <- apply(topic_rate1, 1, which.max)
round(data.frame(�^�l=Z1 %*% 1:k1, ����=topic_allocation1, z=topic_rate1), 3)

topic_rate2 <- SEG2 / rowSums(SEG2)
topic_allocation2 <- apply(topic_rate2, 1, which.max)
round(data.frame(�^�l=Z2 %*% 1:k2, ����=topic_allocation2, z=topic_rate2), 3)