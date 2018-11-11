#####�Ή��g�s�b�N���f��#####
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

####�f�[�^�̐���####
#set.seed(423943)
#�����f�[�^�̐ݒ�
k <- 10   #�g�s�b�N��
d <- 3000   #������
v <- 1000   #��b��
w <- rpois(d, rgamma(d, 75, 0.5))   #1����������̒P�ꐔ
a <- 100   #�⏕�ϐ���
x <- rtpois(d, 22.5, 2, Inf)
f1 <- sum(w)
f2 <- sum(x)

#ID�̐ݒ�
w_id <- rep(1:d, w)
a_id <- rep(1:d, x)

#�p�����[�^�̐ݒ�
alpha0 <- rep(0.2, k)   #�����̃f�B���N�����O���z�̃p�����[�^
alpha1 <- rep(0.15, v)   #�P��̃f�B���N�����O���z�̃p�����[�^
alpha2 <- rep(0.15, a)   #�⏕���̃f�B�N�������O���z�̃p�����[�^

##���f���Ɋ�Â��P��𐶐�
for(rp in 1:1000){
  print(rp)
  
  #�f�B���N�����z����p�����[�^�𐶐�
  thetat <- theta <- extraDistr::rdirichlet(d, alpha0)   #�����̃g�s�b�N���z���f�B���N���������琶��
  phit <- phi <- extraDistr::rdirichlet(k, alpha1)   #�P��̃g�s�b�N���z���f�B���N���������琶��
  lambda <- matrix(0, nrow=d, ncol=k)   #�����Ɋ܂ރg�s�b�N������⏕���̃g�s�b�N�ɂ��邽�߂̊m�����i�[����s��
  omegat <- omega <- rdirichlet(k, alpha2)   #�⏕���̃g�s�b�N���z���f�B�N�����������琶��
  
  #�������z�̗�������f�[�^�𐶐�
  WX <- matrix(0, nrow=d, ncol=v)
  AX <- matrix(0, nrow=d, ncol=a)
  word_list <- list()
  aux_list <- list()
  Z1_list <- list()
  Z2_list <- list()
  
  for(i in 1:d){
    #�����̃g�s�b�N�𐶐�
    z1 <- rmnom(w[i], 1, theta[i, ])   #�����̃g�s�b�N���z�𐶐�
    z1_vec <- as.numeric(z1 %*% 1:k)
    
    #�����̃g�s�b�N���z����P��𐶐�
    word <- rmnom(w[i], 1, phi[z1_vec, ])   #�����̃g�s�b�N����P��𐶐�
    word_vec <- colSums(word)   #�P�ꂲ�Ƃɍ��v����1�s�ɂ܂Ƃ߂�
    WX[i, ] <- word_vec  
    
    #�����̃g�s�b�N���z����⏕�ϐ��𐶐�
    #�����Ő����������g�s�b�N�݂̂�⏕���̃g�s�b�N���z�Ƃ���
    rate <- rep(0, k)
    lambda[i, ] <- colSums(z1) / w[i]

    #�⏕���̃g�s�b�N�𐶐�
    z2 <- rmnom(x[i], 1, lambda[i, ])
    z2_vec <- as.numeric(z2 %*% 1:k)

    #�⏕���̃g�s�b�N����⏕���𐶐�
    aux <- rmnom(x[i], 1, omega[z2_vec, ])
    aux_vec <- colSums(aux)
    AX[i, ] <- aux_vec
    
    #�����g�s�b�N����ѕ⏕���g�s�b�N���i�[
    Z1_list[[i]] <- z1
    Z2_list[[i]] <- z2
    word_list[[i]] <- as.numeric(word %*% 1:v)
    aux_list[[i]] <- as.numeric(aux %*% 1:a)
  }
  if(min(colSums(AX)) > 0 & min(colSums(WX)) > 0){
    break
  }
}

#�f�[�^�s��𐮐��^�s��ɕύX
Z1 <- do.call(rbind, Z1_list)
Z2 <- do.call(rbind, Z2_list)
wd <- unlist(word_list)
ad <- unlist(aux_list)
storage.mode(WX) <- "integer"
storage.mode(AX) <- "integer"


####�g�s�b�N���f������̂��߂̃f�[�^�Ɗ֐��̏���####
##���ꂼ��̕������̒P��̏o������ѕ⏕���̏o�����x�N�g���ɕ��ׂ�
##�C���f�b�N�X���쐬
doc1_list <- list()
doc1_vec <- list()
word_list <- list()
word_vec <- list()
doc2_list <- list()
doc2_vec <- list()
aux_list <- list()
aux_vec <- list()

for(i in 1:d){
  doc1_list[[i]] <- which(w_id==i)
  doc1_vec[[i]] <- rep(1, length(doc1_list[[i]]))
  doc2_list[[i]] <- which(a_id==i)
  doc2_vec[[i]] <- rep(1, length(doc2_list[[i]]))
}
for(j in 1:v){
  word_list[[j]] <- which(wd==j)
  word_vec[[j]] <- rep(1, length(word_list[[j]]))
}
for(j in 1:a){
  aux_list[[j]] <- which(ad==j)
  aux_vec[[j]] <- rep(1, length(aux_list[[j]]))
}
gc(); gc()


####�}���R�t�A�������e�J�����@�őΉ��g�s�b�N���f���𐄒�####
##�P�ꂲ�Ƃɖޓx�ƕ��S�����v�Z����֐�
burden_fr <- function(theta, phi, wd, w, k){
  #���S�W�����v�Z
  Bur <- theta[w, ] * t(phi)[wd, ]   #�ޓx
  Br <- Bur / rowSums(Bur)   #���S��
  r <- colSums(Br) / sum(Br)   #������
  bval <- list(Br=Br, Bur=Bur, r=r)
  return(bval)
}


##�A���S���Y���̐ݒ�
R <- 5000   #�T���v�����O��
keep <- 2   #2���1��̊����ŃT���v�����O���ʂ��i�[
disp <- 10 
burnin <- 1000/keep
iter <- 0

##���O���z�̐ݒ�
#�n�C�p�[�p�����[�^�̎��O���z
alpha01 <- 1
alpha02 <- 1
beta01 <- 0.1
beta02 <- 0.1

##�p�����[�^�̏����l
theta <- rdirichlet(d, rep(1.0, k))  #�����g�s�b�N�̃p�����[�^�̏����l
phi <- rdirichlet(k, rep(0.5, v))   #�P��g�s�b�N�̃p�����[�^�̏����l
omega <- rdirichlet(k, rep(0.5, a))   #�⏕���g�s�b�N�̃p�����[�^�̏����l

##�p�����[�^�̊i�[�p�z��
THETA <- array(0, dim=c(d, k, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
OMEGA <- array(0, dim=c(k, a, R/keep))
W_SEG <- matrix(0, nrow=f1, ncol=k)
A_SEG <- matrix(0, nrow=f2, ncol=k)
storage.mode(W_SEG) <- "integer"
storage.mode(A_SEG) <- "integer"
gc(); gc()

##�ΐ��ޓx�̊�l
LLst1 <- sum(WX %*% log(colSums(WX) / f1))
LLst2 <- sum(AX %*% log(colSums(AX) / f2))
LLst <- LLst1 + LLst2


####�M�u�X�T���v�����O�Ńp�����[�^���T���v�����O####
for(rp in 1:R){
  
  ##�P��g�s�b�N���T���v�����O
  #�P�ꂲ�ƂɃg�s�b�N�̏o�������v�Z
  word_par <- burden_fr(theta, phi, wd, w_id, k)
  word_rate <- word_par$Br

  #�������z����P��g�s�b�N���T���v�����O
  Zi1 <- rmnom(f1, 1, word_rate)
  Zi1_T <- t(Zi1)
  z1_vec <- as.numeric(Zi1 %*% 1:k)
  
  
  ##�P��g�s�b�N�̃p�����[�^���X�V
  #�f�B�N�������z����theta���T���v�����O
  wsum0 <- matrix(0, nrow=d, ncol=k)
  for(i in 1:d){
    wsum0[i, ] <- Zi1_T[, doc1_list[[i]]] %*% doc1_vec[[i]]
  }
  wsum <- wsum0 + alpha01   #�f�B���N�����z�̃p�����[�^
  theta <- extraDistr::rdirichlet(d, wsum)   #�f�B���N�����z����p�����[�^���T���v�����O
  
  #�f�B�N�������z����phi���T���v�����O
  vsum0 <- matrix(0, nrow=k, ncol=v)
  for(j in 1:v){
    vsum0[, j] <- Zi1_T[, word_list[[j]], drop=FALSE] %*% word_vec[[j]]
  }
  vsum <- vsum0 + beta01   #�f�B���N�����z�̃p�����[�^
  phi <- extraDistr::rdirichlet(k, vsum)   #�f�B���N�����z����p�����[�^���T���v�����O

  
  ##�⏕���g�s�b�N���T���v�����O
  #�����������P��g�s�b�N����g�s�b�N���o�m�����v�Z
  theta_aux <- wsum0 / w
  
  #�⏕��񂲂ƂɃg�s�b�N�̏o�������v�Z
  aux_par <- burden_fr(theta_aux, omega, ad, a_id, k)
  aux_rate <- aux_par$Br
  
  #�������z����⏕���g�s�b�N���T���v�����O
  Zi2 <- rmnom(f2, 1, aux_rate)
  Zi2_T <- t(Zi2)
  z2_vec <- as.numeric(Zi2 %*% 1:k)
  
  ##�⏕���g�s�b�N�̃p�����[�^���X�V
  asum0 <- matrix(0, nrow=k, ncol=a)
  for(j in 1:a){
    asum0[, j] <- Zi2_T[, aux_list[[j]], drop=FALSE] %*% aux_vec[[j]]
  }
  asum <- asum0 + beta02   #�f�B���N�����z�̃p�����[�^
  omega <- extraDistr::rdirichlet(k, asum)   #�f�B���N�����z����p�����[�^���T���v�����O
  
  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^���i�[
  if(rp%%keep==0){
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    OMEGA[, , mkeep] <- omega
    
    #�g�s�b�N�����̓o�[���C�����Ԃ𒴂�����i�[����
    if(rp%%keep==0 & rp >= burnin){
      W_SEG <- W_SEG + Zi1
      A_SEG <- A_SEG + Zi2
    }
    
    #�T���v�����O���ʂ��m�F
    if(rp%%disp==0){
      LL <- sum(log(rowSums(word_par$Bur))) + sum(log(rowSums(aux_par$Bur)))
      print(rp)
      print(c(LL, LLst))
      print(round(cbind(theta[1:10, ], thetat[1:10, ]), 3))
      print(round(cbind(omega[, 1:10], omegat[, 1:10]), 3))
    }
  }
}

####�T���v�����O���ʂ̉����Ɨv��####
burnin <- 2000   #�o�[���C������

##�T���v�����O���ʂ̉���
#�����̃g�s�b�N���z�̃T���v�����O����
matplot(t(THETA[1, , ]), type="l", ylab="�p�����[�^", main="����1�̃g�s�b�N���z�̃T���v�����O����")
matplot(t(THETA[2, , ]), type="l", ylab="�p�����[�^", main="����2�̃g�s�b�N���z�̃T���v�����O����")
matplot(t(THETA[3, , ]), type="l", ylab="�p�����[�^", main="����3�̃g�s�b�N���z�̃T���v�����O����")
matplot(t(THETA[4, , ]), type="l", ylab="�p�����[�^", main="����4�̃g�s�b�N���z�̃T���v�����O����")
matplot(t(THETA[5, , ]), type="l", ylab="�p�����[�^", main="����4�̃g�s�b�N���z�̃T���v�����O����")

#�P��̏o���m���̃T���v�����O����
matplot(t(PHI[1, , ]), type="l", ylab="�p�����[�^", main="�g�s�b�N1�̒P��̏o�����̃T���v�����O����")
matplot(t(PHI[3, , ]), type="l", ylab="�p�����[�^", main="�g�s�b�N2�̒P��̏o�����̃T���v�����O����")
matplot(t(PHI[5, , ]), type="l", ylab="�p�����[�^", main="�g�s�b�N3�̒P��̏o�����̃T���v�����O����")
matplot(t(PHI[7, , ]), type="l", ylab="�p�����[�^", main="�g�s�b�N4�̒P��̏o�����̃T���v�����O����")
matplot(t(PHI[9, , ]), type="l", ylab="�p�����[�^", main="�g�s�b�N5�̒P��̏o�����̃T���v�����O����")

#�⏕���̏o���m���̃T���v�����O����
matplot(t(OMEGA[2, , ]), type="l", ylab="�p�����[�^", main="�g�s�b�N1�̕⏕���̏o�����̃p�����[�^�̃T���v�����O����")
matplot(t(OMEGA[4, , ]), type="l", ylab="�p�����[�^", main="�g�s�b�N2�̕⏕���̏o�����̃p�����[�^�̃T���v�����O����")
matplot(t(OMEGA[6, , ]), type="l", ylab="�p�����[�^", main="�g�s�b�N3�̕⏕���̏o�����̃p�����[�^�̃T���v�����O����")
matplot(t(OMEGA[8, , ]), type="l", ylab="�p�����[�^", main="�g�s�b�N4�̕⏕���̏o�����̃p�����[�^�̃T���v�����O����")
matplot(t(OMEGA[10, , ]), type="l", ylab="�p�����[�^", main="�g�s�b�N5�̕⏕���̏o�����̃p�����[�^�̃T���v�����O����")

##�T���v�����O���ʂ̗v�񐄒��
#�g�s�b�N���z�̎��㐄���
topic_mu <- apply(THETA[, , burnin:(R/keep)], c(1, 2), mean)   #�g�s�b�N���z�̎��㕽��
round(cbind(topic_mu, thetat), 3)
round(topic_sd <- apply(THETA[, , burnin:(R/keep)], c(1, 2), sd), 3)   #�g�s�b�N���z�̎���W���΍�

#�P��o���m���̎��㐄���
word_mu <- apply(PHI[, , burnin:(R/keep)], c(1, 2), mean)   #�P��̏o�����̎��㕽��
round(cbind(t(word_mu), t(phit)), 3)

#�⏕���o�����̎��㐄���
tag_mu1 <- apply(OMEGA[, , burnin:(R/keep)], c(1, 2), mean)   #�⏕���̏o�����̎��㕽��
round(cbind(t(tag_mu1), t(omegat)), 3)
