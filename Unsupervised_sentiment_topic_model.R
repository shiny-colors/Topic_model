#####Unsupervised Sentiment topic model#####
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

#set.seed(364023)

####�f�[�^�̔���####
##�f�[�^�̐ݒ�
k1 <- 15   #�g�s�b�N��
k2 <- 3   #�Z���`�����g��
d <- 3000   #������
r <- 5   #�]����
v <- 1000   #��b��
index_v1 <- 1:700   #��ʌ�̌�b����
index_v2 <- 701:v   #�ɐ���̌�b����
v1 <- length(index_v1)
v2 <- length(index_v2)
spl <- split(index_v2, 1:3)
w <- rpois(d, rgamma(d, 75, 0.55))   #�P�ꐔ
f <- sum(w)   #���P�ꐔ

##ID�̐ݒ�
t_id <- c()
d_id <- rep(1:d, w)
for(i in 1:d){
  t_id <- c(t_id, 1:w[i])
}

##�p�����[�^�̐���
#�f�B���N�����z�ƃx�[�^���z�̎��O���z
alpha01 <- rep(0.1, k1) 
alpha02 <- c(0.1, 0.25, 0.2, 0.25, 0.1)
alpha11 <- c(rep(0.1, v1), rep(0.0025, v2))
beta01 <- 100.0; beta02 <- 40.0

#�ɐ����z�̎��O���z
alpha12 <- matrix(0, nrow=r, ncol=v)
alpha12[1, ] <- c(rep(0.005, v1), rep(0.5, length(spl[[1]])), rep(0.025, length(spl[[2]])), rep(0.0025, length(spl[[3]])))
alpha12[2, ] <- c(rep(0.005, v1), rep(0.3, length(spl[[1]])), rep(0.1, length(spl[[2]])), rep(0.025, length(spl[[3]])))
alpha12[3, ] <- c(rep(0.005, v1), rep(0.2, length(spl[[1]])), rep(1.0, length(spl[[2]])), rep(0.2, length(spl[[3]])))
alpha12[4, ] <- c(rep(0.005, v1), rep(0.025, length(spl[[1]])), rep(0.1, length(spl[[2]])), rep(0.3, length(spl[[3]])))
alpha12[5, ] <- c(rep(0.005, v1), rep(0.0025, length(spl[[1]])), rep(0.025, length(spl[[2]])), rep(0.5, length(spl[[3]])))

for(rp in 1:1000){
  print(rp)
  
  #���O���z����p�����[�^�𐶐�
  theta <- thetat <- extraDistr::rdirichlet(d, alpha01)
  ohm <- ohmt <- extraDistr::rdirichlet(d, alpha02)
  phi <- phit <- extraDistr::rdirichlet(k1, alpha11)
  omega <- omegat <- extraDistr::rdirichlet(r, alpha12)
  lambda <- lambdat <- rbeta(d, beta01, beta02)
  
  
  ##���f���Ɋ�Â��f�[�^�𐶐�
  wd_list <- list()
  Z1_list <- Z2_list <- list()
  WX <- matrix(0, nrow=d, ncol=v)
  y <- matrix(0, nrow=d, ncol=r)
  y_vec <- rep(0, d)
  
  for(i in 1:d){
    
    #��ʌꂩ�ɐ��ꂩ�ǂ����𐶐�
    z1 <- rbinom(w[i], 1, lambda[i])
    index <- which(z1==1)
    
    #�]���l�𐶐�
    y[i, ] <- rmnom(1, 1, ohm[i, ])
    y_vec[i] <- as.numeric(y[i, ] %*% 1:r)
    
    #��ʌ�̃g�s�b�N�𐶐�
    z2 <- matrix(0, nrow=w[i], ncol=k1)
    z2[index, ] <- rmnom(sum(z1), 1, theta[i, ])
    z2_vec <- as.numeric(z2 %*% 1:k1)
    
    #�P��𐶐�
    word <- matrix(0, nrow=w[i], ncol=v)
    word[index, ] <- rmnom(sum(z1), 1, phi[z2_vec[index], ])
    word[-index, ] <- rmnom(sum(1-z1), 1, omega[y_vec[i], ])
    
    #�f�[�^���i�[
    WX[i, ] <- colSums(word)
    wd_list[[i]] <- as.numeric(word %*% 1:v)
    Z1_list[[i]] <- z1
    Z2_list[[i]] <- z2
  }
  if(min(colSums(WX)) > 0) break
}

##���X�g��ϊ�
wd <- unlist(wd_list)
Z1 <- unlist(Z1_list)
Z2 <- do.call(rbind, Z2_list)


####�}���R�t�A�������e�J�����@��Unsupervised Sentiment topic model�𐄒�####
##�g�s�b�N���f���̖ޓx�ƕ��S�����`����֐�
burden_fr <- function(theta, phi_T, wd, d_id, k, k_vec){
  #���S�W�����v�Z
  Bur <- theta[d_id, ] * phi_T[wd, ]   #�ޓx
  Bur_sum <- as.numeric(Bur %*% k_vec)
  Br <- Bur / Bur_sum   #���S��
  r <- colSums(Br) / sum(Br)   #������
  bval <- list(Br=Br, Bur=Bur, Bur_sum=Bur_sum, r=r)
  return(bval)
}


##�A���S���Y���̐ݒ�
R <- 5000
keep <- 2  
iter <- 0
burnin <- 1000
disp <- 10

##�C���f�b�N�X�̐ݒ�
doc_list <- list()
doc_vec <- list()
wd_list <- list()
wd_vec <- list()

for(i in 1:d){
  doc_list[[i]] <- which(d_id==i)
  doc_vec[[i]] <- rep(1, length(doc_list[[i]]))
}
for(j in 1:v){
  wd_list[[j]] <- which(wd==j)
  wd_vec[[j]] <- rep(1, length(wd_list[[j]]))
}

##�f�[�^�̐ݒ�
y_vecd <- y_vec[d_id]
y_wd <- y[d_id, ]
k_vec <- rep(1, k1)
r_vec <- rep(1, r)
  
##���O���z�̐ݒ�
#�g�s�b�N���f���̎��O���z
alpha01 <- 0.1
alpha02 <- 0.1

#�x�[�^���z�̎��O���z
beta01 <- 1
beta02 <- 1

##�p�����[�^�̐^�l
theta <- thetat 
ohm <- ohmt 
phi <- phit 
phi_T <- t(phi)
omega <- omegat 
omdga_T <- t(omega)
lambda <- lambdat 

##�����l�̐ݒ�
#�P�ꕪ�z�̃f�B���N�����z�̃p�����[�^
df <- matrix(0, nrow=r, ncol=v)
alpha1 <- matrix(0, nrow=r, ncol=v)
for(i in 1:r){
  df[i, ] <- colSums(WX[y_vec==i, ] > 0)
  alpha1[i, ] <- df[i, ] / quantile(df[i, ], 0.90) + 0.1
}
alpha2 <- as.numeric(colMins(df) > 0) + 0.1

#�p�����[�^�̏����l
lambda <- rep(0.5, d)
theta <- extraDistr::rdirichlet(d, rep(1.0, k1))
omega <- extraDistr::rdirichlet(r, alpha1)
omega_T <- t(omega)
phi <- extraDistr::rdirichlet(k1, alpha2)
phi_T <- t(phi)

##�p�����[�^�̊i�[�p�z��
LAMBDA <- matrix(0, nrow=R/keep, ncol=d)
THETA <- array(0, dim=c(d, k1, R/keep))
PHI <- array(0, dim=c(k1, v, R/keep))
OMEGA <- array(0, dim=c(r, v, R/keep))
SEG1 <- rep(0, f)
SEG2 <- matrix(0, nrow=f, ncol=k1)


##��̑ΐ��ޓx���v�Z
#���j�O�������f���̑ΐ��ޓx
LLst <- sum(WX %*% log(colSums(WX)/f))

#�^�l�ł̑ΐ��ޓx
phit0 <- (phit+10^-100) / rowSums(phit+10^-100); omegat0 <- (omegat+10^-100) / rowSums(omegat+10^-100)
LLbest <- sum(log(Z1*rowSums(thetat[d_id, ]*t(phit0)[wd, ]) + (1-Z1)*rowSums(y[d_id, ]*t(omegat0)[wd, ])))


####�}���R�t�A�������e�J�����@�Ńp�����[�^���T���v�����O####
for(rp in 1:R){

  ##�x���k�[�C���z�����ʌꂩ�ɐ��ꂩ�ǂ������T���v�����O
  #��ʌ�Ƌɐ���̖ޓx
  word_par <- burden_fr(theta, phi_T, wd, d_id, k1, k_vec)
  sentiment_par <- as.numeric((y_wd * omega_T[wd, ]) %*% r_vec)
  
  #���ݕϐ�z�̊����m��
  Lho1 <- lambda[d_id] * word_par$Bur_sum
  Lho2 <- (1-lambda[d_id]) * sentiment_par
  z_rate <- Lho1 / (Lho1+Lho2)
  
  #�x���k�[�C���z������ݕϐ����T���v�����O
  Zi1 <- rbinom(f, 1, z_rate)
  index_z1 <- which(Zi1==1)
  y_wd_T <- t(y_wd * (1-Zi1))
  
  #�x�[�^���z���獬�������T���v�����O
  freq <- tapply(Zi1, d_id, sum)
  lambda <- rbeta(d, freq+beta01, w-freq+beta02)

  ##��ʌ�̃g�s�b�N���T���v�����O
  #���ݕϐ��̊����m��
  z_rate <- word_par$Br[index_z1, ]
  
  #�������z����g�s�b�N���T���v�����O
  Zi2 <- matrix(0, nrow=f, ncol=k1)
  Zi2[index_z1, ] <- rmnom(length(index_z1), 1, z_rate)
  z2_vec <- as.numeric(Zi2 %*% 1:k1)
  Zi2_T <- t(Zi2)
  
  
  ##�p�����[�^���T���v�����O
  #�g�s�b�N���z�̃p�����[�^���T���v�����O
  wsum0 <- matrix(0, nrow=d, ncol=k1)
  for(i in 1:d){
    wsum0[i, ] <- Zi2_T[, doc_list[[i]]] %*% doc_vec[[i]]
  }
  wsum <- wsum0 + alpha01   #�f�B���N�����z�̃p�����[�^
  theta <- extraDistr::rdirichlet(d, wsum)   #�f�B���N�����z����p�����[�^���T���v�����O
  
  #�P�ꕪ�z�̃p�����[�^���T���v�����O
  vf01 <- matrix(0, nrow=k1, ncol=v); vf02 <- matrix(0, nrow=r, ncol=v)
  for(j in 1:v){
    vf01[, j] <- Zi2_T[, wd_list[[j]], drop=FALSE] %*% wd_vec[[j]]   #��ʌ�̕��z
    vf02[, j] <- y_wd_T[, wd_list[[j]], drop=FALSE] %*% wd_vec[[j]]   #�ɐ���̕��z
  }
  vf1 <- vf01 + alpha02; vf2 <- vf02 + alpha02   #�f�B���N�����z�̃p�����[�^
  phi <- extraDistr::rdirichlet(k1, vf1)   #�f�B���N�����z����p�����[�^���T���v�����O
  omega <- extraDistr::rdirichlet(r, vf2)
  phi_T <- t(phi); omega_T <- t(omega)
  
 
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^���i�[
  if(rp%%keep==0){
    #�T���v�����O���ʂ̊i�[
    mkeep <- rp/keep
    LAMBDA[mkeep, ] <- lambda
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    OMEGA[, , mkeep] <- omega
  } 
  
  #�g�s�b�N�����̓o�[���C�����Ԃ𒴂�����i�[����
  if(rp%%keep==0 & rp >= burnin){
    SEG1 <- SEG1 + Zi1
    SEG2 <- SEG2 + Zi2
  }
  
  if(rp%%disp==0){
    #�ΐ��ޓx���v�Z
    LL <- sum(log(Zi1*word_par$Bur_sum + (1-Zi1)*sentiment_par))

    #�T���v�����O���ʂ��m�F
    print(rp)
    print(c(LL, LLbest, LLst))
    print(round(c(mean(lambda), mean(lambdat)), 3))
    print(round(cbind(phi[, c(index_v1[1:5], index_v2[1:5])], phit[, c(index_v1[1:5], index_v2[1:5])]), 3))
    print(round(cbind(omega[, c(index_v1[1:5], index_v2[1:5])], omegat[, c(index_v1[1:5], index_v2[1:5])]), 3))
  }
}


####�T���v�����O���ʂ̗v��Ɖ���####
burnin <- 2000/keep
RS <- R/keep

##�T���v�����O���ʂ̉���
#�g�s�b�N���z�̉���
matplot(t(THETA[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA[10, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA[100, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA[1000, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA[2000, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")

#�P�ꕪ�z�̉���
matplot(t(PHI[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI[5, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI[10, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI[15, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(OMEGA[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(OMEGA[3, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(OMEGA[5, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")


##�T���v�����O���ʂ̎��㕪�z
#�x�[�^���z�̎��㕽��
round(cbind(colMeans(LAMBDA[burnin:RS, ]), lambdat), 3)

#�g�s�b�N���z�̎��㕽��
round(cbind(apply(THETA[, , burnin:RS], c(1, 2), mean), thetat), 3)

#�P�ꕪ�z�̎��㕽��
PHI0 <- round(cbind(t(apply(PHI[, , burnin:RS], c(1, 2), mean)), t(phit)), 3)
OMEGA0 <- round(cbind(t(apply(OMEGA[, , burnin:RS], c(1, 2), mean)), t(omegat)), 3)

##���ݕϐ��̃T���v�����O���ʂ̎��㕪�z
seg1_rate <- SEG1 / max(SEG1)
seg2_rate <- SEG2 / rowSums(SEG2)
seg1_rate[is.nan(seg1_rate)] <- 0; seg2_rate[is.nan(seg2_rate)] <- 0

#�g�s�b�N�������ʂ��r
round(cbind(SEG1, seg1_rate, Z1), 3)
round(cbind(rowSums(SEG2), seg2_rate, Z2), 3)

