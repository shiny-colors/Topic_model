#####Hidden Topic Marcov Model#####
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

#set.seed(5723)

####�f�[�^�̔���####
k <- 10   #�g�s�b�N��
d <- 2000   #������
v <- 500   #��b��
s <- rpois(d, 15)   #���͐�
s[s < 5] <- ceiling(runif(sum(s < 5), 5, 10))
a <- sum(s)   #�����͐�
w <- rpois(a, 12)   #���͂�����̒P�ꐔ
w[w < 5] <- ceiling(runif(sum(w < 5), 5, 10))
f <- sum(w)   #���P�ꐔ

#����ID�̐ݒ�
u_id <- rep(1:d, s)
t_id <- c()
for(i in 1:d){t_id <- c(t_id, 1:s[i])}
words <- as.numeric(tapply(w, u_id, sum))

#���͋�؂�̃x�N�g�����쐬
w_id <- rep(1:d, words)
x_vec <- rep(0, f)
x_vec[c(1, cumsum(w[-a])+1)] <- 1

##�p�����[�^�̐ݒ�
#�f�B���N�����z�̃p�����[�^
alpha01 <- rep(0.15, k)
alpha11 <- rep(0.1, v) 

#�p�����[�^�𐶐�
for(rp in 1:100){
  theta0 <- thetat0 <- extraDistr::rdirichlet(d, alpha01)
  phi <- phit <- extraDistr::rdirichlet(k, alpha11)
  beta0 <- beta0t <- -2.75
  beta1 <- beta1t <- 2.0
  
  ##���f���ɂ��ƂÂ��P��𐶐�����
  wd_list <- list()
  ID_list <- list()
  td_list <- list()
  Z1_list <- list()
  Z2_list <- list()
  
  for(i in 1:d){
    if(i%%100==0){
      print(i)
    }
    
    freq <- words[i]
    index_w <- which(w_id==i)
    z1_vec <- rep(0, freq)
    z2_vec <- rep(0, freq)
      
    for(j in 1:freq){
      if(j==1){
        #�����̊J�n�g�s�b�N�𐶐�
        z2 <- rmnom(1, 1, theta0[i, ])
        z2_vec[j] <- as.numeric(z2 %*% 1:k)
        
      } else {
        
        #�g�s�b�N�̐؊����𐶐�
        z2_vec[j-1]
        logit <- beta0 + beta1 * x_vec[index_w[j]]
        pr <- exp(logit)/(1+exp(logit))
        z1_vec[j] <- rbinom(1, 1, pr)
        
        #������2�P��ڈȍ~�̃g�s�b�N�𐶐�
        if(z1_vec[j]==1){
          #�g�s�b�N�؊������������ꍇ�V���ȃg�s�b�N�𐶐�
          z2 <- rmnom(1, 1, theta0[i, ])
          z2_vec[j] <- as.numeric(z2 %*% 1:k)
          
        } else {
          
          #�g�s�b�N�؊������Ȃ������ꍇ1�O�Ɠ����g�s�b�N�𐶐�
          z2_vec[j] <- z2_vec[j-1]
        }
      }
    }
    #�g�s�b�N�ɂ��ƂÂ��P��𐶐�
    wn <- rmnom(freq, 1, phi[z2_vec, ])
    wd_list[[i]] <- as.numeric(wn %*% 1:v)
    
    #�p�����[�^���i�[
    ID_list[[i]] <- rep(i, freq)
    td_list[[i]] <- 1:freq
    Z1_list[[i]] <- z1_vec
    Z2_list[[i]] <- z2_vec
  }
  
  #���X�g���x�N�g���ϊ�
  ID_d <- unlist(ID_list)
  td_d <- unlist(td_list)
  wd <- unlist(wd_list)
  z1 <- unlist(Z1_list)
  z2 <- unlist(Z2_list)
  WX <- sparseMatrix(i=1:f, j=wd, x=rep(1, f), dims=c(f, v))   #�X�p�[�X�s����쐬
  theta <- thetat <- matrix(as.numeric(table(ID_d, z2) / words), nrow=d, ncol=k)
  if(length(which(colSums(WX)==0))==0) break
}
sum(colSums(WX)==0)

##�C���f�b�N�X���쐬
doc_list <- list()
word_list <- list()
for(i in 1:d){doc_list[[i]] <- which(ID_d==i)}
for(i in 1:v){word_list[[i]] <- which(wd==i)}


####�}���R�t�A�������e�J�����@��HTM���f���𐄒�####
##���W�X�e�B�b�N��A���f���̑ΐ��ޓx�֐�
#���W�X�e�B�b�N��A���f���̑ΐ��ޓx���`
loglike <- function(alpha, beta, x, y){
  
  #�ޓx���`���č��v����
  logit <- alpha + x * beta 
  p <- exp(logit) / (1 + exp(logit))
  LLS <- y*log(p) + (1-y)*log(1-p)  
  LL <- sum(LLS)
  return(LL)
}

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


####LDA�ŏ����l�𐶐�####
##�A���S���Y���̐ݒ�
R0 <- 1000
keep0 <- 2  
iter0 <- 0
burnin0 <- 200/keep0
disp0 <- 10

#�P��g�s�b�N�P�ʂ̃p�����[�^�̏����l
word_data <- matrix(0, nrow=d, ncol=v)
for(i in 1:d){
  word_data[i, ] <- colSums(WX[doc_list[[i]], ])
}
tf0 <- colSums(word_data)/sum(word_data)
idf0 <- log(d / colSums(word_data > 0))
theta <- extraDistr::rdirichlet(d, rep(0.3, k))   #�����P�ʂ̃g�s�b�N�̏����l
phi0 <- extraDistr::rdirichlet(k, tf0*10) + 0.001   #�����P�ʂ̏o���m���̏����l
phi <- phi0 / rowSums(phi0)

#�n�C�p�[�p�����[�^�̐ݒ�
alpha01 <- 1
beta0 <- 0.5

#�p�����[�^�̊i�[�p�z��
THETA0 <- array(0, dim=c(d, k, R/keep))
PHI0 <- array(0, dim=c(k, v, R/keep))
SEG0 <- matrix(0, nrow=f, ncol=k)
storage.mode(SEG0) <- "integer"

#�ΐ��ޓx�̊�l
LLst <- sum(WX %*% log(colSums(WX)/sum(WX)))


#�M�u�X�T���v�����O�Ńp�����[�^���T���v�����O
for(rp in 1:R0){
  
  ##�P��g�s�b�N���T���v�����O
  #�P�ꂲ�ƂɃg�s�b�N�̏o�������v�Z
  word_par <- burden_fr(theta, phi, wd, words, k)
  word_rate <- word_par$Br
  
  #�������z����P��g�s�b�N���T���v�����O
  Zi <- rmnom(f, 1, word_rate)   
  z_vec <- Zi %*% 1:k
  
  ##�P��g�s�b�N�̃p�����[�^���X�V
  #�f�B�N�������z����theta���T���v�����O
  wsum0 <- matrix(0, nrow=d, ncol=k)
  for(i in 1:d){
    wsum0[i, ] <- colSums(Zi[doc_list[[i]], ])
  }
  wsum <- wsum0 + alpha01 
  theta <- extraDistr::rdirichlet(d, wsum)
  
  #�f�B�N�������z����phi���T���v�����O
  vf0 <- matrix(0, nrow=k, ncol=v)
  for(j in 1:v){
    vf0[, j] <- colSums(Zi[word_list[[j]], , drop=FALSE])
  }
  vf <- vf0 + beta0
  phi <- extraDistr::rdirichlet(k, vf)
  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^���i�[
  if(rp%%keep0==0){
    #�T���v�����O���ʂ̊i�[
    mkeep <- rp/keep0
    THETA0[, , mkeep] <- theta
    PHI0[, , mkeep] <- phi
    
    #�g�s�b�N�����̓o�[���C�����Ԃ𒴂�����i�[����
    if(rp%%keep0==0 & rp >= burnin0){
      SEG0 <- SEG0 + Zi
    }
    
    #�T���v�����O���ʂ��m�F
    if(rp%%disp==0){
      print(rp)
      print(c(sum(log(rowSums(word_par$Bur))), LLst))
      print(round(cbind(theta[1:10, ], thetat[1:10, ]), 3))
      print(round(cbind(phi[, 1:10], phit[, 1:10]), 3))
    }
  }
}


####HTM���f����MCMC�A���S���Y���̐ݒ�####
##�A���S���Y���̐ݒ�
R <- 10000
keep <- 2  
iter <- 0
burnin <- 1000/keep
disp <- 10

##�p�����[�^�̐^�l
theta <- thetat
phi <- phit
beta0 <- beta0t
beta1 <- beta1t
r0 <- exp(beta0)/(1+exp(beta0))
r1 <- exp(beta0+beta1)/(1+exp(beta0+beta1))
r <- cbind(r0, r1)

##LDA����HTM���f���̏����l��ݒ�
theta <- apply(THETA0[, , burnin0:(R0/keep0)], c(1, 2), mean)
phi <- apply(PHI0[, , burnin0:(R0/keep0)], c(1, 2), mean)
beta0 <- -2.5
beta1 <- 1.5
r0 <- exp(beta0)/(1+exp(beta0))
r1 <- exp(beta0+beta1)/(1+exp(beta0+beta1))
r <- cbind(r0, r1)

##���O���z�̐ݒ�
#�n�C�p�[�p�����[�^�̎��O���z
alpha01 <- 0.01 
beta01 <- 1
betas <- rep(0, 2)  #��A�W���̏����l
B0 <- 0.01*diag(2)
rw <- 0.025

##�p�����[�^�̊i�[�p�z��
THETA <- array(0, dim=c(d, k, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
BETA <- matrix(0, nrow=R/keep, 2)
SEG1 <- rep(0, f)
SEG2 <- matrix(0, nrow=f, ncol=k)
storage.mode(SEG1) <- "integer"
storage.mode(SEG2) <- "integer"


##MCMC����p�z��
max_word <- max(words)
index_t11 <- which(td_d==1)
index_t21 <- list()
index_t22 <- list()
for(j in 2:max_word){
  index_t21[[j]] <- which(td_d==j)-1
  index_t22[[j]] <- which(td_d==j)
}


####�M�u�X�T���v�����O��HTM���f���̃p�����[�^���T���v�����O####
for(rp in 1:R){
  
  #�P�ꂲ�Ƃ̃g�s�b�N�ޓx�ƃg�s�b�N�̌��𐶐�
  word_par <- burden_fr(theta, phi, wd, words, k)   
  Li01 <- word_par$Bur   #�g�s�b�N���f���̊��Җޓx
  Zi02 <- rmnom(f, 1, word_par$Br)     #�������z����g�s�b�N�𐶐�
  z02 <- as.numeric(Zi02 %*% 1:k)

  #�}���R�t���ڃ��f���̖ޓx�Ɗ����m���𒀎��I�ɐ���
  Zi1 <- rep(0, f)
  z1_rate <- rep(0, f)
  rf02 <- rep(0, f)
  
  for(j in 2:max_word){
    
    #�f�[�^�̐ݒ�
    index <- index_t22[[j]]
    x0 <- x_vec[index]
    Li01_obz <- Li01[index, , drop=FALSE]
    Zi02_obz <-  Zi02[index_t21[[j]], ]
    
    #�}���R�t�؊����m���𐄒�
    Li11 <- (1-r[x0+1]) * rowSums(Li01_obz * Zi02_obz)
    Li12 <- r[x0+1] * rowSums(Li01_obz * (1-Zi02_obz))
    z1_rate[index] <- Li12 / (Li11+Li12)

    #�x���k�[�C���z����؊����ϐ��𐶐�
    Zi1[index] <- rbinom(length(index), 1, z1_rate[index])
    index_z1 <- which(Zi1[index]==0)
    if(length(index_z1)==0) next
    
    if(length(index)!=1){
      Zi02[index, ][index_z1, ] <- Zi02[index_t21[[j]], , drop=FALSE][index_z1, ]
    } else {
      Zi02[index, ] <- Zi02[index_t21[[j]], ]
    }
  }
  Zi2 <- Zi02
  z2_vec <- as.numeric(Zi2 %*% 1:k)


  ##MH�@�ō������̉�A�p�����[�^���T���v�����O
  #�f�[�^�̐ݒ�
  y <- Zi1[-index_t11]
  x <- x_vec[-index_t11]

  #beta�̃T���v�����O
  betad <- c(beta0, beta1)
  betan <- betad + rw * rnorm(2)   #�V����beta�������_���E�H�[�N�ŃT���v�����O
  
  #�ΐ��ޓx�Ƒΐ����O���z�̌v�Z
  lognew <- loglike(betan[1], betan[2], x, y)
  logold <- loglike(betad[1], betad[2], x, y)
  logpnew <- lndMvn(betan, betas, B0)
  logpold <- lndMvn(betad, betas, B0)
  
  
  #MH�T���v�����O
  alpha <- min(1, exp(lognew + logpnew - logold - logpold))
  if(alpha == "NAN") alpha <- -1
  
  #��l�����𔭐�
  u <- runif(1)
  
  #u < alpha�Ȃ�V����beta���̑�
  if(u < alpha){
    beta0 <- betan[1]
    beta1 <- betan[2]
    
    #�����łȂ��Ȃ�beta���X�V���Ȃ�
  } else {
    beta0 <- betad[1]
    beta1 <- betad[2]
  }
  
  #�؊����m���̍��������X�V
  r0 <- exp(beta0) / (1+exp(beta0))
  r1 <- exp(beta0+beta1) / (1+exp(beta0+beta1))
  r <- cbind(r0, r1)

  ##�g�s�b�N���f���̃p�����[�^���T���v�����O
  #�g�s�b�N���ztheta���T���v�����O
  wsum0 <- matrix(0, nrow=d, ncol=k)
  for(i in 1:d){
    wsum0[i, ] <- colSums(Zi2[doc_list[[i]], ])
  }
  wsum <- wsum0 + beta01
  theta <- extraDistr::rdirichlet(d, wsum)
  
  #�P�ꕪ�zphi���T���v�����O
  vf0 <- matrix(0, nrow=k, ncol=v)
  for(j in 1:v){
    vf0[, j] <- colSums(Zi2[word_list[[j]], , drop=FALSE])
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
    BETA[mkeep, ] <- c(beta0, beta1)
    
    #�g�s�b�N�����̓o�[���C�����Ԃ𒴂�����i�[����
    if(mkeep >= burnin & rp%%keep==0){
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
    }
    
    #�T���v�����O���ʂ��m�F
    if(rp%%disp==0){
      print(rp)
      print(c(sum(log(rowSums(word_par$Bur))), LLst))
      print(round(cbind(theta[1:6, ], thetat[1:6, ]), 3))
      print(round(cbind(phi[, 1:10], phit[, 1:10]), 3))
      round(print(c(beta0, beta1, r)), 3)
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


##�T���v�����O���ʂ̗v�񐄒��
#�g�s�b�N���z�̎��㐄���
topic_mu <- apply(THETA[, , burnin:(R/keep)], c(1, 2), mean)   #�g�s�b�N���z�̎��㕽��
round(cbind(topic_mu, thetat), 3)
round(topic_sd <- apply(THETA[, , burnin:(R/keep)], c(1, 2), sd), 3)   #�g�s�b�N���z�̎���W���΍�

#�P��o���m���̎��㐄���
word_mu <- apply(PHI[, , burnin:(R/keep)], c(1, 2), mean)   #�P��̏o�����̎��㕽��
word <- round(t(rbind(word_mu, phit)), 3)
colnames(word) <- 1:ncol(word)
word

##�g�s�b�N�̎��㕪�z�̗v��
round(cbind(z1, seg1_mu <- SEG1 / length(burnin:RS)), 3)
round(cbind(z2, seg2_mu <- SEG2 / rowSums(SEG2)), 3)




