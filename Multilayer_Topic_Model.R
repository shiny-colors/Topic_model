#####Multilayer Topic Model#####
options(warn=2)
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
#�f�[�^�̐ݒ�
k0 <- 5   #��ʊK�w�̃g�s�b�N��
k1 <- 10   #���ʊK�w�̃g�s�b�N��
k_len <- k0
d <- 3000   #������
v <- 700   #��b��
w <- rpois(d, rgamma(d, 100, 0.55))
f <- sum(w)

#����ID�̐ݒ�
d_id <- rep(1:d, w)
t_id <- c()
for(i in 1:d){t_id <- c(t_id, 1:w[i])}

#��b�g�s�b�N�̃C���f�b�N�X
for(j in 1:100){
  v_index <- c(1, round(sort(runif(k_len-1, 100, v-100))), v)
  x <- v_index[2:length(v_index)] - v_index[1:(length(v_index)-1)]
  if(min(x) > 100) break
}

##�p�����[�^�̐ݒ�
#�f�B���N�����z�̃p�����[�^
alpha01 <- rep(0.25, k0)
alpha11 <- rep(0.2, k1)

beta01 <- matrix(0.3, nrow=k0, ncol=k_len)
diag(beta01) <- 2.0
beta02 <- list()
for(j in 1:k_len){
  beta <- rep(0.0001, v)
  beta[v_index[j]:v_index[j+1]] <- 0.1
  beta02[[j]] <- beta
}

#�p�����[�^�𐶐�
theta01 <- thetat01 <- extraDistr::rdirichlet(d, alpha01)
phi01 <- phit01 <- extraDistr::rdirichlet(k0, beta01)
theta02 <- thetat02 <- extraDistr::rdirichlet(d, alpha11)
phi02 <- phit02 <- list()
for(j in 1:k_len){
  par <- extraDistr::rdirichlet(k1, beta02[[j]]) + runif(v, 10^-100, 10^-50)
  phi02[[j]] <- phit02[[j]] <- par / rowSums(par)
}


##�K�w�g�s�b�N���f������f�[�^�𐶐�
Z1_list <- list()
Z2_list <- list()
S_list <- list()
word_list <- list()
WX_list <- list()

for(i in 1:d){
  ##��ʊK�w�̃f�[�^�𐶐�
  #��ʊK�w�̃g�s�b�N�𐶐�
  z1 <- rmnom(w[i], 1, theta01[i, ])
  z1_vec <- as.numeric(z1 %*% 1:k0)
  
  #���ʊK�w�̊����𐶐�
  s <- rmnom(w[i], 1, phi01[z1_vec, ])
  s_vec <- as.numeric(s %*% 1:k_len)
  
  ##���ʊK�w�̃f�[�^�𐶐�
  z2_vec <- c()
  words <- matrix(0, nrow=w[i], ncol=v)
  for(j in 1:w[i]){
    #���ʊK�w�̃g�s�b�N�𐶐�
    z2 <- rmnom(1, 1, theta02[i,])
    z2_vec <- c(z2_vec, as.numeric(z2 %*% 1:k1))

    #�g�s�b�N����P��𐶐�
    words[j, ] <- rmnom(1, 1, phi02[[s_vec[j]]][z2_vec[j], ])
  }
  
  ##�f�[�^���i�[
  Z1_list[[i]] <- z1_vec
  Z2_list[[i]] <- z2_vec
  S_list[[i]] <- s_vec
  word_list[[i]] <- words
  WX_list[[i]] <- colSums(words)
}

##���X�g��ϊ�
Z1 <- unlist(Z1_list)
Z2 <- unlist(Z2_list)
s <- unlist(S_list); s_freq <- table(s)
Data <- do.call(rbind, word_list)
wd <- as.numeric(Data %*% 1:v)
WX <- do.call(rbind, WX_list)
storage.mode(WX) <- "integer"
sparse_data <- as(Data, "CsparseMatrix")
rm(Data)
hist(colSums(WX), xlab="�p�x", main="�P��̕p�x���z", col="grey", breaks=25)
summary(colSums(WX))

##�C���f�b�N�X���쐬
doc_list <- list()
wd_list <- list()
for(i in 1:d){doc_list[[i]] <- which(d_id==i)}
for(i in 1:v){wd_list[[i]] <- which(wd==i)}


####�}���R�t�A�������e�J�����@�ŊK�w�g�s�b�N���f���𐄒�####
##�P�ꂲ�Ƃɖޓx�ƕ��S�����v�Z����֐�
burden_fr <- function(theta, phi, wd, w, k){
  Bur <-  matrix(0, nrow=length(wd), ncol=k)   #���S�W���̊i�[�p
  for(j in 1:k){
    #���S�W�����v�Z
    Bi <- rep(theta[, j], w) * phi[j, wd]   #�ޓx
    Bur[, j] <- Bi   
  }
  Br <- Bur / rowSums(Bur)   #���S���̌v�Z
  bval <- list(Br=Br, Bur=Bur)
  return(bval)
}


##�A���S���Y���̐ݒ�
R <- 5000
keep <- 2  
iter <- 0
burnin <- 1000/keep
disp <- 10

##�p�����[�^�̎��O���z
alpha01 <- 10
alpha02 <- 1
beta01 <- 0.1
beta02 <- 0.0001

##�p�����[�^�̐^�l
theta1 <- thetat01
theta2 <- thetat02
phi1 <- phit01
phi2 <- phit02
r <- phi1[Z1, ]

##�p�����[�^�̏����l
#�f�B���N�����z�̃p�����[�^
beta1 <- matrix(0.3, nrow=k0, ncol=k_len)
diag(beta1) <- 1.0
beta2 <- list()
for(j in 1:k_len){ 
  beta <- rep(0.1, v)
  beta[v_index[j]:v_index[j+1]] <- 2.0
  beta2[[j]] <- beta
}

#�p�����[�^�𐶐�
theta1 <- extraDistr::rdirichlet(d, rep(0.25, k0))
theta2 <- extraDistr::rdirichlet(d, alpha11 <- rep(0.2, k1))
phi1 <- extraDistr::rdirichlet(k0, beta1)
phi2 <- list()
for(j in 1:k_len){
  par <- extraDistr::rdirichlet(k1, beta2[[j]]) + runif(v, 10^-100, 10^-50)
  phi2[[j]] <- par / rowSums(par)
}
r <- matrix(1/k_len, nrow=f, ncol=k_len)


##�p�����[�^�̊i�[�p�z��
THETA1 <- array(0, dim=c(d, k0, R/keep))
THETA2 <- array(0, dim=c(d, k1, R/keep))
PHI1 <- array(0, dim=c(k0, k_len, R/keep))
PHI2 <- array(0, dim=c(k_len*k1, v, R/keep))
S <- matrix(0, nrow=f, ncol=k_len)
SEG1 <- matrix(0, nrow=f, ncol=k0)
SEG2 <- matrix(0, nrow=f, ncol=k1)
storage.mode(S) <- "integer"
storage.mode(SEG1) <- "integer"
storage.mode(SEG2) <- "integer"


##�f�[�^�̐ݒ�
index_allocation <- matrix(1:(k_len*k1), ncol=k_len)
vec1_list <- list()
vec2_list <- list()
for(i in 1:d){
  vec1_list[[i]] <- rep(1, w[i])
}
for(j in 1:v){
  vec2_list[[j]] <- rep(1,  sum(WX[, j]))
}
wsum01 <- matrix(0, nrow=d, ncol=k0)
wsum02 <- matrix(0, nrow=d, ncol=k1)
vf01 <- matrix(0, nrow=k0, k_len)
vf02 <- matrix(0, nrow=k1*k_len, v)


##�ΐ��ޓx�̊�l
#���j�O�������f���̑ΐ��ޓx
par0 <- colSums(WX) / f + beta02
par <- par0 / sum(par0)
LLst <- sum(WX %*% log(par))


####�}���R�t�A�������e�J�����@�Ńp�����[�^���T���v�����O####
for(rp in 1:R){
  
  ##���ʊK�w�̐��ݕϐ��̊������T���v�����O
  #��ʃg�s�b�N�̖ޓx�𐄒�
  LLi <- matrix(0, nrow=f, ncol=k_len)
  word_par <- list()
  for(j in 1:k_len){
    word_par[[j]] <- burden_fr(theta2, phi2[[j]], wd, w, k1)
    LLi[, j] <- rowSums(word_par[[j]]$Bur)
  }

  #���ݕϐ��̊������T���v�����O
  allocation_par <- r * LLi
  s_rate <- allocation_par / rowSums(allocation_par)   #���ݕϐ��̊����m��
  Si <- rmnom(f, 1, s_rate)   #�������z������ݕϐ����T���v�����O
  s_vec <- as.numeric(Si %*% 1:k_len)
  
  
  ##��ʃg�s�b�N���T���v�����O
  #��ʃg�s�b�N�̖ޓx�Ɗ����m���𐄒�
  word_par1 <- burden_fr(theta1, phi1, s_vec, w, k0)
  z1_rate <- word_par1$Br
  
  #�������z�����ʃg�s�b�N���T���v�����O
  Zi1 <- rmnom(f, 1, z1_rate)
  z1_vec <- as.numeric(Zi1 %*% 1:k0)
  r <- phi1[z1_vec, ]   #�������̍X�V

  
  ##���ʃg�s�b�N���T���v�����O
  Zi2 <- matrix(0, nrow=f, ncol=k1)
  index_s <- list()
  for(j in 1:k_len){
    index <- which(Si[, j]==1)
    index_s[[j]] <- index
    z2_rate <- word_par[[j]]$Br[index, ] 
    Zi2[index, ] <- rmnom(length(index), 1, z2_rate)   
  }


  ##�g�s�b�N���z���T���v�����O
  #�f�B���N�����z�̃p�����[�^�𐄒�
  for(i in 1:d){
    wsum01[i, ] <- vec1_list[[i]] %*% Zi1[doc_list[[i]], ] + alpha01
    wsum02[i, ] <- vec1_list[[i]] %*% Zi2[doc_list[[i]], ] + alpha02
  }
  #�f�B���N�����z����g�s�b�N���z���T���v�����O
  theta1 <- extraDistr::rdirichlet(d, wsum01)
  theta2 <- extraDistr::rdirichlet(d, wsum02)

  
  ##�P�ꕪ�z���T���v�����O
  #�f�B���N�����z�̃p�����[�^�𐄒�
  n <- colSums(Si)
  for(j in 1:k_len){vf01[, j] <- rep(1, n[j]) %*% Zi1[index_s[[j]], ]}
  for(j in 1:v){vf02[, j] <- as.numeric(t(Zi2[wd_list[[j]], , drop=FALSE]) %*% Si[wd_list[[j]], ])}
  vf1 <- vf01 + beta01
  vf2 <- vf02 + beta02
  
  
  #�f�B���N�����z����P�ꕪ�z���T���v�����O
  phi1 <- extraDistr::rdirichlet(k0, vf1)
  phi0 <- extraDistr::rdirichlet(k_len*k1, vf2)
  phi2 <- list()
  for(j in 1:k_len){phi2[[j]] <- phi0[index_allocation[, j], ]}

  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^���i�[
  if(rp%%keep==0){
    #�T���v�����O���ʂ̊i�[
    mkeep <- rp/keep
    THETA1[, , mkeep] <- theta1
    THETA2[, , mkeep] <- theta2
    PHI1[, , mkeep] <- phi1
    PHI2[, , mkeep] <- phi0
     
    #�g�s�b�N�����̓o�[���C�����Ԃ𒴂�����i�[����
    if(rp%%keep==0 & rp >= burnin){
      S <- S + Si
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
    }
    
    #�T���v�����O���ʂ��m�F
    if(rp%%disp==0){
      print(rp)
      LL <- sum(log(rowSums(LLi * Si)))
      print(c(LL, LLst))
      print(round(c(exp(-LL / f), exp(-LLst / f)), 3))
      print(round(cbind(phi1, phit01), 3))
      print(round(cbind(theta2[1:10, ], thetat02[1:10, ]), 3))
      print(rbind(si_freq=colSums(Si), s_freq))
    }
  }
}

####�T���v�����O���ʂ̉����Ɨv��####
burnin <- 2000/keep   #�o�[���C������
RS <- R/keep

##�T���v�����O���ʂ̉���
#��ʊK�w�̃g�s�b�N���z�̉���
matplot(t(THETA1[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA1[2, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA1[3, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")

#��ʊK�w�̊����m���̕��z�̉���
matplot(t(PHI1[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI1[2, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI1[3, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")

#���ʊK�w�̃g�s�b�N���z�̉���
matplot(t(THETA2[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA2[2, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA2[3, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA2[4, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA2[5, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")

#���ʊK�w�̒P�ꕪ�z�̉���
matplot(t(PHI2[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI2[10, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI2[20, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI2[30, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI2[40, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI2[50, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")


##�T���v�����O���ʂ̗v�񐄒��
#��ʃg�s�b�N���z�̎��㐄���
topic_mu1 <- apply(THETA1[, , burnin:(R/keep)], c(1, 2), mean)   #�g�s�b�N���z�̎��㕽��
round(cbind(topic_mu1, thetat01), 3)
round(topic_sd1 <- apply(THETA1[, , burnin:(R/keep)], c(1, 2), sd), 3)   #�g�s�b�N���z�̎���W���΍�

#��ʊK�w�̊����m���̎��㐄���
word_mu1 <- apply(PHI1[, , burnin:(R/keep)], c(1, 2), mean)   #�P��̏o�����̎��㕽��
round(t(rbind(word_mu1, phit01)), 3)

#���ʃg�s�b�N���z�̎��㐄���
topic_mu2 <- apply(THETA2[, , burnin:(R/keep)], c(1, 2), mean)   #�g�s�b�N���z�̎��㕽��
round(cbind(topic_mu2, thetat02), 3)
round(topic_sd2 <- apply(THETA2[, , burnin:(R/keep)], c(1, 2), sd), 3)   #�g�s�b�N���z�̎���W���΍�

#���ʊK�w�̒P��o���m���̎��㐄���
word_mu2 <- apply(PHI2[, , burnin:(R/keep)], c(1, 2), mean)   #�P��̏o�����̎��㕽��
round(t(word_mu2), 3)


##������ꂽ�g�s�b�N�̎��㕪�z
round(cbind(SEG1 / rowSums(SEG1), Z1), 3)
round(cbind(SEG2 / rowSums(SEG2), apply(SEG1 / rowSums(SEG1), 1, which.max), Z1, Z2), 3)

