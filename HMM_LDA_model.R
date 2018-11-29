#####HMM-LDA���f��#####
options(warn=0)
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
library(Matrix)
library(data.table)
library(bayesm)
library(HMM)
library(extraDistr)
library(reshape2)
library(dplyr)
library(plyr)
library(ggplot2)
'%!in%' <- function(a,b) ! a %in% b

#set.seed(2506787)

####�f�[�^�̔���####
##�f�[�^�̐ݒ�
k1 <- 8   #syntax��
k2 <- 15   #�g�s�b�N��
d <- 3000   #������
v1 <- 700   #�g�s�b�N�Ɋ֌W�̂����b��
v2 <- 500   #�g�s�b�N�ȊO��syntax�Ɋ֌W�̂����b��
v <- v1 + v2   #����b��
w <- rpois(d, rgamma(d, 40, 0.15))   #����������̒P�ꐔ
f <- sum(w)   #���P�ꐔ
vec_k1 <- rep(1, k1)
vec_k2 <- rep(1, k2)

##ID�̐ݒ�
d_id <- rep(1:d, w)
t_id <- as.numeric(unlist(tapply(1:f, d_id, rank)))

##�p�����[�^��ݒ�
#�f�B���N�����z�̃p�����[�^
alpha01 <- seq(2.5, 0.5, length=k1)
alpha02 <- matrix(0.5, nrow=k1, ncol=k1); alpha02[-k1, k1] <- 8.0; alpha02[k1, k1] <- 3.0
alpha11 <- rep(0.15, k2)
alpha21 <- c(rep(0.05, v1), rep(0.00025, v2))

#syntax�̎��O���z
alloc <- as.numeric(rmnom(v2, 1, extraDistr::rdirichlet(k1-1, rep(5.0, k1-1))) %*% 1:(k1-1))
alpha22 <- cbind(matrix(0.0001, nrow=k1-1, ncol=v1), matrix(0.0025, nrow=k1-1, ncol=v2))
for(j in 1:(k1-1)){
  index <- which(alloc==j) + v1
  alpha22[j, index] <- 2.5
}

##���f���Ɋ�Â��P��𐶐�
rp <- 0
repeat {
  rp <- rp + 1
  print(rp)
  
  #�f�B���N�����z���p�����[�^�𐶐�
  pi1 <- pit1 <- as.numeric(extraDistr::rdirichlet(1, alpha01))
  pi2 <- pit2 <- extraDistr::rdirichlet(k1, alpha02)
  theta <- thetat <- extraDistr::rdirichlet(d, alpha11)
  phi <- extraDistr::rdirichlet(k2, alpha21)
  psi <- extraDistr::rdirichlet(k1-1, alpha22)
  
  #�P��o���m�����Ⴂ�g�s�b�N�����ւ���
  index1 <- which(colMaxs(psi) < (k1*10)/f); index1 <- index1[index1 > v1]
  index2 <- which(colMaxs(phi) < (k2*10)/f); index2 <- index2[index2 <= v1]
  for(j in 1:length(index1)){
    psi[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, rep(1.0, k1-1))) %*% 1:(k1-1)), index1[j]] <- (k1*10)/f
  }
  for(j in 1:length(index2)){
    phi[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, rep(1.0, k2))) %*% 1:k2), index2[j]] <- (k2*10)/f
  }
  psit <- psi
  phit <- phi
  
  ##HMM-LDA���f���Ɋ�Â��P��𐶐�
  wd_list <- Z1_list <- Z2_list <- list()
  WX <- matrix(0, nrow=d, ncol=v)
  
  for(i in 1:d){
    z1_vec <- rep(0, w[i])
    z2_vec <- rep(0, w[i])
    words <- matrix(0, nrow=w[i], ncol=v)
    
    for(j in 1:w[i]){
      if(j==1){
        #�����̐擪�P���syntax�𐶐�
        z1 <- rmnom(1, 1, pi1)
        z1_vec[j] <- as.numeric(z1 %*% 1:k1)
      } else {
        #�擪�ȍ~�̓}���R�t���ڂɊ�Â�syntax�𐶐�
        z1 <- rmnom(1, 1, pi2[z1_vec[j-1], ])
        z1_vec[j] <- as.numeric(z1 %*% 1:k1)
      }
    }
    sum(z1_vec==k1)
    
    #�g�s�b�N���z�𐶐�
    index_topic <- which(z1_vec==k1)
    z2 <- rmnom(length(index_topic), 1, theta[i, ])
    z2_vec[index_topic] <- as.numeric(z2 %*% 1:k2)
    
    #�g�s�b�N���z�Ɋ�Â��P��𐶐�
    words[-index_topic, ] <- rmnom(w[i]-length(index_topic), 1, psi[z1_vec[-index_topic], ])   #syntax�Ɋ֘A����P��
    words[index_topic, ] <- rmnom(length(index_topic), 1, phi[z2_vec[index_topic], ])   #�g�s�b�N�Ɋ֘A����P��
    

    #�f�[�^���i�[
    wd_list[[i]] <- as.numeric(words %*% 1:v)
    WX[i, ] <- colSums(words) 
    Z1_list[[i]] <- z1_vec
    Z2_list[[i]] <- z2_vec
  }
  if(min(colSums(WX)) > 0){
    break
  }
}

#���X�g�`�����x�N�g���`���ɕϊ�
z1 <- unlist(Z1_list); Z1 <- matrix(as.numeric(table(1:f, z1)), nrow=f, ncol=k1)
z2 <- unlist(Z2_list)
wd <- unlist(wd_list)
word_data <- sparseMatrix(1:f, wd, x=rep(1, f), dims=c(f, v))   #�P��x�N�g�����s��
word_dt <- t(word_data)

#�C���f�b�N�X���쐬
d_list <- word_list <- list()
for(i in 1:d){d_list[[i]] <- which(d_id==i)}
for(i in 1:v){word_list[[i]] <- which(wd==i)}


####�}���R�t�A�������e�J�����@��HMM-LDA���f���𐄒�####
##�P�ꂲ�Ƃɖޓx�ƕ��S�����v�Z����֐�
burden_fr <- function(theta, phi, wd, w, k, vec_k){
  #���S�W�����v�Z
  Bur <- theta[w, ] * t(phi)[wd, ]   #�ޓx
  Br <- Bur / as.numeric(Bur %*% vec_k)   #���S��
  bval <- list(Br=Br, Bur=Bur)
  return(bval)
}


####�M�u�X�T���v���[��HMM-LDA���f���𐄒�####
##�A���S���Y���̐ݒ�
R <- 3000
keep <- 2  
iter <- 0
burnin <- 500
disp <- 10
er <- 0.00005

##�C���f�b�N�X�ƃf�[�^�̐ݒ�
#�f�[�^�̐ݒ�
d_data <- sparseMatrix(1:f, d_id, x=rep(1, f), dims=c(f, d))   #�����x�N�g�����s��
d_dt <- t(d_data)

#�擪�ƌ���̃C���f�b�N�X���쐬
max_word <- max(t_id)
index_t11 <- which(t_id==1)
index_t12 <- rep(0, d)
for(i in 1:d){
  index_t12[i] <- max(d_list[[i]])
}

#���Ԃ̃C���f�b�N�X���쐬
index_list_t21 <- index_list_t22 <- list()
for(j in 2:max_word){
  index_list_t21[[j]] <- which(t_id==j)-1
  index_list_t22[[j]] <- which(t_id==j)
}
index_t21 <- sort(unlist(index_list_t21))
index_t22 <- sort(unlist(index_list_t22))


##���O���z�̐ݒ�
alpha01 <- 0.01 
alpha02 <- 0.01
beta01 <- 0.01
beta02 <- 0.01

##�p�����[�^�̐^�l
#�p�����[�^�̏����l
pi1 <- as.numeric(pit1)
pi2 <- pit2
theta <- thetat
phi <- phit
psi <- psit

#HMM�̐��ݕϐ��̏����l
z1_vec <- z1
Zi1 <- matrix(as.numeric(table(1:f, z1)), nrow=f, ncol=k1) %*% 1:k1

##�����l��ݒ�
#�p�����[�^�̏����l
pi1 <- as.numeric(extraDistr::rdirichlet(1, rep(2.0, k1)))
pi2 <- extraDistr::rdirichlet(k1, rep(2.0, k1))
theta <- extraDistr::rdirichlet(d, rep(2.0, k2))
phi <- extraDistr::rdirichlet(k2, rep(1.0, v))
psi <- extraDistr::rdirichlet(k1-1, rep(1.0, v))

#HMM�̐��ݕϐ��̏����l
Zi1 <- rmnom(f, 1, rep(1/k1, k1))
z1_vec <- as.numeric(Zi1 %*% 1:k1)


##�p�����[�^�̊i�[�p�z��
PI1 <- matrix(0, nrow=R/keep, ncol=k1)
PI2 <- array(0, dim=c(k1, k1, R/keep))
THETA <- array(0, dim=c(d, k2, R/keep))
PHI <- array(0, dim=c(k2, v, R/keep))
PSI <- array(0, dim=c(k1-1, v, R/keep))
SEG1 <- matrix(0, nrow=f, ncol=k1)
SEG2 <- matrix(0, nrow=f, ncol=k2)
storage.mode(SEG1) <- "integer"
storage.mode(SEG2) <- "integer"


##�ΐ��ޓx�̊�l
#���j�O�������f���̑ΐ��ޓx
LLst <- sum(WX %*% log(colSums(WX)/sum(WX)))   

#�x�X�g�ȑΐ��ޓx
LL <- c()
LL1 <- sum(log(as.numeric((Z1[z1!=k1, -k1] * t(psit)[wd[z1!=k1], ]) %*% rep(1, k1-1))))
LL2 <- sum(log(as.numeric(((thetat[d_id, ] * t(phit)[wd, ])[z1==k1, ]) %*% vec_k2)))
LLbest <- LL1 + LL2


####�M�u�X�T���v�����O�Ńp�����[�^���T���v�����O####
for(rp in 1:R){
  
  ##�P�ꂲ�Ƃ̖ޓx�ƍ�������ݒ�
  #syntax�ƃg�s�b�N���f���̖ޓx
  Li01 <- t(psi)[wd, ]   #syntax���Ƃ̖ޓx
  word_par <- burden_fr(theta, phi, wd, d_id, k2, vec_k2) 
  Li02 <- rowSums(word_par$Bur)   #�g�s�b�N���f���̊��Җޓx
  Li0 <- cbind(Li01, Li02)   #�ޓx�̌���
  
  #HMM�̍�����
  pi_dt1 <- pi_dt2 <- matrix(1, nrow=f, ncol=k1)
  pi_dt1[index_t11, ] <- matrix(pi1, nrow=d, ncol=k1, byrow=T)   #�����̐擪�ƌ���̍�����
  pi_dt1[index_t22, ] <- pi2[z1_vec[index_t21], ]   #1�P��O�̍�����
  pi_dt2[index_t21, ] <- t(pi2)[z1_vec[index_t22], ]   #1�P���̍�����
  

  ##�������z����HMM�̐��ݕϐ����T���v�����O
  #���ݕϐ��̊����m��
  Li <- pi_dt1 * pi_dt2 * Li0   #�������z
  z1_rate <- Li / as.numeric(Li %*% vec_k1)   #�����m��
  
  #���ݕϐ����T���v�����O
  Zi1 <- rmnom(f, 1, z1_rate)
  z1_vec <- as.numeric(Zi1 %*% 1:k1)
  n1 <- sum(Zi1[, 1:(k1-1)]); n2 <- sum(Zi1[, k1])
  index_topic <- which(Zi1[, k1]==1)
  
  ##HMM�̃p�����[�^���T���v�����O
  #�f�B���N�����z���琄�ڊm�����T���v�����O
  rf11 <- colSums(Zi1[index_t11, ]) + alpha01
  rf12 <- t(Zi1[index_t21, ]) %*% Zi1[index_t22, ] + alpha02
  pi1 <- as.numeric(extraDistr::rdirichlet(1, rf11))
  pi2 <- extraDistr::rdirichlet(k1, rf12)

  
  #�f�B�N�������z����syntax�̃p�����[�^���T���v�����O
  Zi1_syntax <- Zi1[-index_topic, ]
  df <- t(as.matrix(word_dt[, -index_topic] %*% Zi1_syntax)) + alpha02
  psi <- extraDistr::rdirichlet(k1-1, df)
  
  
  ##�P��̃g�s�b�N���T���v�����O
  Zi2 <- matrix(0, nrow=f, ncol=k2)
  Zi2[index_topic, ] <- rmnom(n2, 1, word_par$Br[index_topic, ])   #�������z���g�s�b�N���T���v�����O
  z2_vec <- as.numeric(Zi2 %*% 1:k2)

  
  ##�g�s�b�N���f���̃p�����[�^���T���v�����O
  #�g�s�b�N���z���T���v�����O
  Zi2_topic <- Zi2[index_topic, ]
  wsum <- as.matrix(d_dt[, index_topic] %*% Zi2_topic) + beta01
  theta <- extraDistr::rdirichlet(d, wsum)
  
  #�P�ꕪ�z���T���v�����O
  vf <- t(as.matrix(word_dt[, index_topic] %*% Zi2_topic)) + beta02
  phi <- extraDistr::rdirichlet(k2, vf)
  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^���i�[
  if(rp%%keep==0){
    #�T���v�����O���ʂ̊i�[
    mkeep <- rp/keep
    PI1[mkeep, ] <- pi1
    PI2[, , mkeep] <- pi2
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    PSI[, , mkeep] <- psi
    
    #�g�s�b�N�����̓o�[���C�����Ԃ𒴂�����i�[����
    if(mkeep >= burnin & rp%%keep==0){
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
    }
    
    if(rp%%disp==0){
      #�ΐ��ޓx���v�Z
      LL1 <- sum(log(as.numeric((Zi1[-index_topic, -k1] * t(psi)[wd[-index_topic], ]) %*% rep(1, k1-1))))
      LL2 <- sum(log(as.numeric(((theta[d_id, ] * t(phi)[wd, ])[index_topic, ]) %*% vec_k2)))
      LL <- c(LL, LL1 + LL2)
      
      #�T���v�����O���ʂ�\��
      print(rp)
      print(c(LL1+LL2, LLbest, LLst))
      print(round(cbind(pi2, pit2), 3))
      print(round(cbind(psi[, (v1-4):(v1+5)], psit[, (v1-4):(v1+5)]), 3))
    }
  }
}

####�T���v�����O���ʂ̉����Ɨv��####
burnin <- 1000/keep   #�o�[���C������
RS <- R/keep

##�T���v�����O���ʂ̉���
#�����̃g�s�b�N���z�̃T���v�����O����
matplot(t(THETA[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA[100, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA[1000, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA[2000, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")

#�P��̏o���m���̃T���v�����O����
matplot(t(PHI[, 1, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N1�̒P��̏o�����̃T���v�����O����")
matplot(t(PHI[, 200, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N2�̒P��̏o�����̃T���v�����O����")
matplot(t(PHI[, 400, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N3�̒P��̏o�����̃T���v�����O����")
matplot(t(PHI[, 500, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N4�̒P��̏o�����̃T���v�����O����")
matplot(t(PSI[, 1, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N1�̒P��̏o�����̃T���v�����O����")
matplot(t(PSI[, 200, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N2�̒P��̏o�����̃T���v�����O����")
matplot(t(PSI[, 400, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N3�̒P��̏o�����̃T���v�����O����")
matplot(t(PSI[, 500, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N4�̒P��̏o�����̃T���v�����O����")


##�T���v�����O���ʂ̗v�񐄒��
#�g�s�b�N���z�̎��㐄���
topic_mu <- apply(THETA[, , burnin:(R/keep)], c(1, 2), mean)   #�g�s�b�N���z�̎��㕽��
round(cbind(topic_mu, thetat), 3)
round(topic_sd <- apply(THETA[, , burnin:(R/keep)], c(1, 2), sd), 3)   #�g�s�b�N���z�̎���W���΍�

#�P��o���m���̎��㐄���
word_mu1 <- apply(PHI[, , burnin:(R/keep)], c(1, 2), mean)   #�P��̏o�����̎��㕽��
word1 <- round(t(rbind(word_mu1, phit)), 3)
word_mu2 <- apply(OMEGA[, , burnin:(R/keep)], c(1, 2), mean)   #�P��̏o�����̎��㕽��
word2 <- round(t(rbind(word_mu2, omegat)), 3)
word <- round(t(rbind(word_mu1, word_mu2, phit, omegat)), 3)
colnames(word) <- 1:ncol(word)

word_mu3 <- apply(GAMMA[burnin:(R/keep), ], 2, mean)   #�P��̏o�����̎��㕽��
round(rbind(word_mu3, gamma=gammat), 3)


##�g�s�b�N�̎��㕪�z�̗v��
round(seg1_mu <- SEG1 / rowSums(SEG1), 3)
round(seg2_mu <- SEG2 / rowSums(SEG2), 3)