#####Syntax Latent Dirichlet Allocation���f��#####
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
library(Matrix)
library(bayesm)
library(extraDistr)
library(reshape2)
library(plyr)
library(dplyr)
library(ggplot2)


#set.seed(21437)

####�f�[�^�̔���####
#set.seed(423943)
#�f�[�^�̐ݒ�
k <- 10   #�g�s�b�N��
d <- 2000   #������
v1 <- 300   #�g�s�b�N�Ɋ֌W�̂����b��
v2 <- 100   #�g�s�b�N�Ɋ֌W�̂Ȃ���b��
v <- v1 + v2   #����b��
w <- rpois(d, rgamma(d, 60, 0.50))   #1����������̒P�ꐔ
f <- sum(w)


#�p�����[�^�̐ݒ�
alpha0 <- rep(0.15, k)   #�����̃f�B���N�����O���z�̃p�����[�^
alpha1 <- c(rep(0.4, v1), rep(0.0075, v2))   #�g�s�b�N�Ɋ֌W�̂���P��̃f�B���N�����O���z�̃p�����[�^
alpha2 <- c(rep(0.1, v1), rep(10, v2))   #��ʌ�̃f�B���N�����O���z�̃p�����[�^

#�f�B���N�������̔���
thetat <- theta <- extraDistr::rdirichlet(d, alpha0)   #�����̃g�s�b�N���z���f�B���N���������甭��
phit <- phi <- extraDistr::rdirichlet(k, alpha1)   #�P��̃g�s�b�N���z���f�B���N���������甭��
gammat <- gamma <- extraDistr::rdirichlet(1, alpha2)   #��ʌ�̒P�ꕪ�z���f�B���N���������甭��
betat <- beta <- rbeta(d, 20, 15)


#�������z�̗�������f�[�^�𔭐�
WX <- matrix(0, nrow=d, ncol=v)
y_list <- list()
Z_list <- list()

for(i in 1:d){
  #�����̃g�s�b�N�𐶐�
  z <- rmnom(w[i], 1, theta[i, ])   #�����̃g�s�b�N���z�𔭐�
  z_vec <- as.numeric(z %*% c(1:k))   #�g�s�b�N�������x�N�g����
  
  #��ʌꂩ�ǂ����𐶐�
  y0 <- rbinom(w[i], 1, beta)
  index <- which(y0==1)
  
  #�g�s�b�N����P��𐶐�
  wn1 <- rmnom(length(index), 1, phi[z_vec[index], ])   #�����̃g�s�b�N����P��𐶐�
  wn2 <- rmnom(1, w[i]-length(index), gamma)   #��ʌ�𐶐�
  wdn <- colSums(wn1) + colSums(wn2)   #�P�ꂲ�Ƃɍ��v����1�s�ɂ܂Ƃ߂�
  WX[i, ] <- wdn
  Z_list[[i]] <- z
  y_list[[i]] <- y0
  print(i)
}

#���X�g�`����ϊ�
Z <- do.call(rbind, Z_list)
y_vec <- unlist(y_list)


####�g�s�b�N���f������̂��߂̃f�[�^�Ɗ֐��̏���####
##�f�[�^����pID���쐬
ID_list <- list()
wd_list <- list()

#���l���Ƃɋ��lID����ђP��ID���쐬
for(i in 1:nrow(WX)){
  print(i)
  
  #�P���ID�x�N�g�����쐬
  ID_list[[i]] <- rep(i, w[i])
  num1 <- (WX[i, ] > 0) * (1:v)
  num2 <- which(num1 > 0)
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


####�}���R�t�A�������e�J�����@�őΉ��g�s�b�N���f���𐄒�####
##�P�ꂲ�Ƃɖޓx�ƕ��S�����v�Z����֐�
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
disp <- 20
burnin <- 1000/keep

##���O���z�̐ݒ�
#�n�C�p�[�p�����[�^�̎��O���z
alpha01 <- 1.0
beta01 <- 0.5
beta02 <- 0.5
beta03 <- c(f/50, f/50)


##�p�����[�^�̏����l
#tfidf�ŏ����l��ݒ�
tf <- WX/rowSums(WX)
idf1 <- log(nrow(WX)/colSums(WX > 0))
idf2 <- log(nrow(WX)/colSums(WX==0))

theta <- extraDistr::rdirichlet(d, rep(1, k))   #�����g�s�b�N�̃p�����[�^�̏����l
phi <- extraDistr::rdirichlet(k, idf1*10)   #�P��g�s�b�N�̃p�����[�^�̏����l
gamma <- extraDistr::rdirichlet(1, idf2*100)   #��ʌ�̃p�����[�^�̏����l
r <- 0.5   #�������̏����l

##�p�����[�^�̊i�[�p�z��
THETA <- array(0, dim=c(d, k, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
GAMMA <- matrix(0, nrow=R/keep, v)
SEG <- matrix(0, nrow=f, ncol=k)
Y <- rep(0, f)
storage.mode(SEG) <- "integer"
gc(); gc()

##MCMC����p�z��
wsum0 <- matrix(0, nrow=d, ncol=k)
vf0 <- matrix(0, nrow=k, ncol=v)
df0 <- rep(0, v)


####�M�u�X�T���v�����O�Ńp�����[�^���T���v�����O####
for(rp in 1:R){
  
  ##�P��g�s�b�N���T���v�����O
  #�g�s�b�N�̏o�����Ɩޓx�𐄒�
  par <- burden_fr(theta, phi, wd, w, k)
  LH1 <- par$Bur
  word_rate <- par$Br
  
  #�g�s�b�N�Ɋ֌W�̂̏o�����𐄒�
  LH2 <- gamma[wd]
  
  
  ##��ʌꂩ�ǂ������T���v�����O
  Bur1 <- r * rowSums(LH1)
  Bur2 <- (1-r) * LH2
  switch_rate <- Bur1 / (Bur1 + Bur2)
  y <- rbinom(f, 1, switch_rate) 
  index_y <- which(y==1)
  
  #�x�[�^���z���獬�������X�V
  par <- sum(y)
  r <- rbeta(1, par+beta03[1], f-par+beta03[2])
  
  ##�������z����P��g�s�b�N���T���v�����O
  Zi <- rmnom(f, 1, word_rate)   
  Zi[-index_y, ] <- 0
  z_vec <- as.numeric(Zi %*% 1:k)
  
  
  ##�p�����[�^���T���v�����O
  #�g�s�b�N���z���f�B�N�������z����T���v�����O
  wsum0 <- matrix(0, nrow=d, ncol=k)
  for(i in 1:d){
    wsum0[i, ] <- colSums(Zi[doc_list[[i]], ])
  }
  wsum <- wsum0 + alpha01
  theta <- extraDistr::rdirichlet(d, wsum)
  
  #�g�s�b�N��̕��z���f�B�N�������z����T���v�����O
  vf0 <- matrix(0, nrow=k, ncol=v)
  for(j in 1:v){
    vf0[, j] <- colSums(Zi[word_list[[j]], ])
  }
  vf <- vf0 + beta01
  phi <- extraDistr::rdirichlet(k, vf)
  
  #��ʌ�̕��z���f�B�N�������z����T���v�����O
  y0 <- 1-y
  df0 <- rep(0, v)
  for(j in 1:v){
    df0[j] <- sum(y0[word_list[[j]]])
  }
  df <- df0 + beta02
  gamma <- extraDistr::rdirichlet(1, df)
  
  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^���i�[
  if(rp%%keep==0){
    #�T���v�����O���ʂ̊i�[
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    GAMMA[mkeep, ] <- gamma 
    
    #�g�s�b�N�����̓o�[���C�����Ԃ𒴂�����i�[����
    if(rp%%keep==0 & rp >= burnin){
      SEG <- SEG + Zi
      Y <- Y + y
    }
    
    #�T���v�����O���ʂ��m�F
    if(rp%%disp==0){
      print(rp)
      print(c(mean(y), mean(y_vec)))
      print(round(cbind(theta[1:10, ], thetat[1:10, ]), 3))
      print(round(cbind(phi[, 296:305], phit[, 296:305]), 3))
      print(round(rbind(gamma[296:305], gammat[296:305]), 3))
    }
  }
}


####�T���v�����O���ʂ̉����Ɨv��####
burnin <- 1000/keep   #�o�[���C������
RS <- R/keep

##�T���v�����O���ʂ̉���
#�����̃g�s�b�N���z�̃T���v�����O����
matplot(t(THETA[1, , ]), type="l", ylab="�p�����[�^", main="����1�̃g�s�b�N���z�̃T���v�����O����")
matplot(t(THETA[2, , ]), type="l", ylab="�p�����[�^", main="����2�̃g�s�b�N���z�̃T���v�����O����")
matplot(t(THETA[3, , ]), type="l", ylab="�p�����[�^", main="����3�̃g�s�b�N���z�̃T���v�����O����")
matplot(t(THETA[4, , ]), type="l", ylab="�p�����[�^", main="����4�̃g�s�b�N���z�̃T���v�����O����")

#�P��̏o���m���̃T���v�����O����
matplot(t(PHI[1, 296:305, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N1�̒P��̏o�����̃T���v�����O����")
matplot(t(PHI[2, 296:305, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N2�̒P��̏o�����̃T���v�����O����")
matplot(t(PHI[3, 296:305, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N3�̒P��̏o�����̃T���v�����O����")
matplot(t(PHI[4, 296:305, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N4�̒P��̏o�����̃T���v�����O����")

#��ʌ�̏o���m���̃T���v�����O����
matplot(t(PHI[1, 286:295, ]), type="l", ylab="�p�����[�^", main="�P��̏o�����̃T���v�����O����")
matplot(t(PHI[2, 296:305, ]), type="l", ylab="�p�����[�^", main="�P��̏o�����̃T���v�����O����")
matplot(t(PHI[3, 306:315, ]), type="l", ylab="�p�����[�^", main="�P��̏o�����̃T���v�����O����")


##�T���v�����O���ʂ̗v�񐄒��
#�g�s�b�N���z�̎��㐄���
topic_mu <- apply(THETA[, , burnin:(R/keep)], c(1, 2), mean)   #�g�s�b�N���z�̎��㕽��
round(cbind(topic_mu, thetat), 3)
round(topic_sd <- apply(THETA[, , burnin:(R/keep)], c(1, 2), sd), 3)   #�g�s�b�N���z�̎���W���΍�

#�P��o���m���̎��㐄���
word_mu1 <- apply(PHI[, , burnin:(R/keep)], c(1, 2), mean)   #�P��̏o�����̎��㕽��
round(rbind(word_mu1, phit)[, 276:325], 3)

word_mu2 <- apply(GAMMA[burnin:(R/keep), ], 2, mean)   #�P��̏o�����̎��㕽��
round(rbind(word_mu, gamma=gammat), 3)
