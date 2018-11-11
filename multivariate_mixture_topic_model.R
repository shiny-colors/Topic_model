#####���ϗʍ���LDA���f��#####
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

#set.seed(2578)

####�f�[�^�̔���####
##�f�[�^�̐ݒ�
k1 <- 7   #���͂̃Z�O�����g��
k2 <- 7   #�����P�ʂ̂̃g�s�b�N��
k3 <- 8   #�Z�O�����g�P�ʂ̃g�s�b�N�� 
d <- 2000   #������
v1 <- 200   #�����W���S�̂ɋ��ʂ����b��
v2 <- 300   #�����P�̂ɋ��ʂ����b�� 
v <- v1 + v2
s <- rpois(d, 11.5)   #���͐�
s[s < 5] <- ceiling(runif(sum(s < 5), 5, 10))
a <- sum(s)   #�����͐�
w1 <- rpois(a, 5.5)   
w1[w1 < 2] <- ceiling(runif(sum(w1 < 2), 2, 5))
w2 <- rpois(a, 10.5)   #���͂�����̒P�ꐔ
w2[w2 < 3] <- ceiling(runif(sum(w2 < 3), 3, 10))
f1 <- sum(w1)   #���P�ꐔ
f2 <- sum(w2) 
f <- f1 + f2

#����ID�̐ݒ�
u_id <- rep(1:d, s)
t_id <- c()
for(i in 1:d){t_id <- c(t_id, 1:s[i])}
words1 <- as.numeric(tapply(w1, u_id, sum))
words2 <- as.numeric(tapply(w2, u_id, sum))
words <- words1 + words2

##�p�����[�^��ݒ�
#�f�B���N�����z�̃p�����[�^
alpha01 <- seq(3.0, 0.2, length=k1*5)[((1:(k1*5))%%5)==0]
alpha02 <- matrix(0.3, nrow=k1, ncol=k1)
diag(alpha02) <- 2.5
alpha03 <- rep(0.25, k2)
alpha04 <- rep(0.3, k3)
alpha11 <- rep(0.1, v1)
alpha12 <- rep(0.4, v2)
alpha13 <- rep(0.05, v2)

#�f�B���N�����z���p�����[�^�𐶐�
omegat <- omega <- extraDistr::rdirichlet(1, alpha01)   #�����̐擪�g�s�b�N
gammat <- gamma <- extraDistr::rdirichlet(k1, alpha02)   #�����Z�O�����g�P�ʂ̃g�s�b�N���z
thetat1 <- theta1 <- extraDistr::rdirichlet(d, alpha03)   #�����P�ʂ̃g�s�b�N���z
thetat2 <- theta2 <- extraDistr::rdirichlet(k1, alpha04)   #�Z�O�����g�P�ʂ̃g�s�b�N���z
psit <- psi <- extraDistr::rdirichlet(k1, alpha11)
phit <- phi <- extraDistr::rdirichlet(k2, alpha12)
tau <- taut <- extraDistr::rdirichlet(k3, alpha13)
betat <- beta <- rbeta(d, 25, 17.5)


##���͂��ƂɒP��𐶐�����
WX1 <- matrix(0, nrow=a, ncol=v1)
WX2 <- matrix(0, nrow=a, ncol=v2)
y_list <- list()
Z1_list <- list()
Z2_list <- list()
Z3_list <- list()

for(i in 1:d){
  if(i%%100==0){
    print(i)
  }
  z1_vec <- rep(0, s[i])
  
  for(j in 1:s[i]){
    ##�P�ꂲ�Ƃɐ����ߒ�������
    index <- which(u_id==i)[j]
    
    ##���͂��ƂɃg�s�b�N�𐶐�
    freq1 <- w1[index]
    freq2 <- w2[index]
    
    if(j==1){
      z1 <- rmnom(1, 1, omega)
      z1_vec[j] <- as.numeric(z1 %*% 1:k1)
    } else {
      z1 <- rmnom(1, 1, gamma[z1_vec[j-1], ])
      z1_vec[j] <- as.numeric(z1 %*% 1:k1)
    }
    
    ##�P�ꂲ�ƂɃg�s�b�N�𐶐�
    y <- rbinom(freq2, 1, beta[i])
    z2_vec <- rep(0, freq2)
    z3_vec <- rep(0, freq2)
    sum(1-y)
    
    if(sum(y) > 0){
      z2 <- rmnom(sum(y), 1, theta1[i, ])
      z2_vec[y==1] <- as.numeric(z2 %*% 1:k2)
    }
    if(sum(1-y) > 0){
      z3 <- rmnom(sum(1-y), 1, theta2[z1_vec[j], ])
      z3_vec[y==0] <- as.numeric(z3 %*% 1:k3)
    }
    
    ##�g�s�b�N���z�Ɋ�Â��P��𐶐�
    #���͒P�ʂł̒P��𐶐�
    wn1 <- rep(0, v1)
    wn1 <- colSums(rmnom(freq1, 1, psi[z1_vec[j], ]))
    
    #�P��P�ʂł̒P��𐶐�
    wn2 <- rep(0, v2)
    if(sum(y) > 0){
      wn2 <- colSums(rmnom(sum(y), 1, phi[z2_vec[z2_vec!=0], ]))
    }
    wn3 <- rep(0, v2)
    if(sum(1-y) > 0){
      wn3 <- colSums(rmnom(sum(1-y), 1, tau[z3_vec[z3_vec!=0], ]))
    }
    
    #���������P����i�[
    WX1[index, ] <- wn1
    WX2[index, ] <- wn2 + wn3
    if(is.na(sum(WX2[index, ]))==TRUE) {break}
    
    #�p�����[�^���i�[
    Z2_list[[index]] <- z2_vec
    Z3_list[[index]] <- z3_vec
    y_list[[index]] <- y
  }
  Z1_list[[i]] <- z1_vec
}

#�P��o�����Ȃ��P����폜
index_zeros <- which(colSums(WX1)==0)
WX1 <- WX1[, -index_zeros]
psi <- psit <- psi[, -index_zeros]
v1 <- ncol(WX1)

#���X�g�`����ϊ�
Y <- unlist(y_list)
z1 <- unlist(Z1_list)
z2 <- unlist(Z2_list)
z3 <- unlist(Z3_list)
WX <- cbind(WX1, WX2)


####�g�s�b�N���f������̂��߂̃f�[�^�Ɗ֐��̏���####
##���ꂼ��̕������̒P��̏o������ѕ⏕���̏o�����x�N�g���ɕ��ׂ�
##�f�[�^����pID���쐬
ID_list <- list()
td1_list <- list()
td2_list <- list()
wd_list <- list()

#����id���Ƃɕ���id����ђP��id���쐬
for(i in 1:a){
  
  #����ID���L�^
  ID_list[[i]] <- rep(u_id[i], w2[i])
  td1_list[[i]] <- rep(i, w2[i])
  td2_list[[i]] <- rep(t_id[i], w2[i])
  
  #�P��ID���L�^
  num1 <- WX2[i, ] * 1:v2
  num2 <- which(num1 > 0)
  W1 <- WX2[i, (WX2[i, ] > 0)]
  number <- rep(num2, W1)
  wd_list[[i]] <- number
}

#���X�g���x�N�g���ɕϊ�
ID_d <- unlist(ID_list)
td1_d <- unlist(td1_list)
td2_d <- unlist(td2_list)
wd <- unlist(wd_list)

##�C���f�b�N�X���쐬
doc_list <- list()
id_list <- list()
sent_list <- list()
word_list <- list()
for(i in 1:d){doc_list[[i]] <- which(ID_d==i)}
for(i in 1:d){id_list[[i]] <- which(u_id==i)}
for(i in 1:a){sent_list[[i]] <- which(td1_d==i)}
for(i in 1:v2){word_list[[i]] <- which(wd==i)}


####�}���R�t�A�������e�J�����@��LDA���f���𐄒�####
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

#�ΐ��ޓx�̖ڕW�l
LLst1 <- sum(dmnom(WX1, rowSums(WX1), colSums(WX1)/sum(WX1), log=TRUE))
LLst2 <- sum(WX2 %*% log(colSums(WX2)/sum(WX2)))
LLst <- LLst1 + LLst2


##�A���S���Y���̐ݒ�
R <- 10000
keep <- 2  
iter <- 0
burnin <- 2000/keep
disp <- 10

##���O���z�̐ݒ�
#�n�C�p�[�p�����[�^�̎��O���z
alpha01 <- 1 
alpha02 <- 1
alpha03 <- 1
beta01 <- 0.5
beta02 <- 0.5
beta03 <- 0.5

##�p�����[�^�̏����l
theta1 <- thetat1
theta2 <- thetat2
rt0 <- r0 <- as.numeric(omega)
rt1 <- r1 <- gammat
rt2 <- r2 <- beta[ID_d]
psi <- psit
phi <- phit
tau <- taut

#tfidf�ŏ����l��ݒ�
tf11 <- colMeans(WX1)*100+1
tf21 <- colMeans(WX2)*10
idf21 <- log(nrow(WX2)/colSums(WX2 > 0))
idf22 <- log(nrow(WX2)/colSums(WX2==0))

#�P��g�s�b�N�P�ʂ̃p�����[�^�̏����l
theta1 <- extraDistr::rdirichlet(d, rep(1, k2))   #�����P�ʂ̃g�s�b�N�̏����l
theta2 <- extraDistr::rdirichlet(k1, rep(1, k3))   #�Z�O�����g�P�ʂ̃g�s�b�N�̏����l
psi <- extraDistr::rdirichlet(k1, tf11)   #���͒P�ʂ̒P��o���m���̏����l
phi <- extraDistr::rdirichlet(k2, tf21)   #�����P�ʂ̒P��o���m���̏����l
tau <- extraDistr::rdirichlet(k3, tf21)   #�Z�O�����g�P�ʂ̒P��o���m���̏����l
r0 <- rep(1/k1, k1)
par <- matrix(0.3, nrow=k1, ncol=k1)
diag(par) <- 2.0
r1 <- extraDistr::rdirichlet(k1, par)
r2 <- rep(0.5, f2)


##�p�����[�^�̊i�[�p�z��
THETA1 <- array(0, dim=c(d, k2, R/keep))
THETA2 <- array(0, dim=c(k1, k3, R/keep))
R0 <- matrix(0, nrow=R/keep, ncol=k1)
R1 <- array(0, dim=c(k1, k1, R/keep))
PSI <- array(0, dim=c(k1, v1, R/keep))
PHI <- array(0, dim=c(k2, v2, R/keep))
TAU <- array(0, dim=c(k3, v2, R/keep))
SEG1 <- matrix(0, nrow=a, ncol=k1)
SEG2 <- matrix(0, nrow=f2, ncol=k2)
SEG3 <- matrix(0, nrow=f2, ncol=k3)
storage.mode(SEG1) <- "integer"
storage.mode(SEG2) <- "integer"
storage.mode(SEG3) <- "integer"

##MCMC����p�z��
max_time <- max(t_id)
index_t11 <- which(t_id==1)
index_t21 <- list()
index_t22 <- list()
for(j in 2:max_time){
  index_t21[[j]] <- which(t_id==j)-1
  index_t22[[j]] <- which(t_id==j)
}
wx1_const <- lfactorial(w1) - rowSums(lfactorial(WX1))   #�������z�̖��x�֐��̑ΐ��ޓx�̒萔

####�M�u�X�T���v�����O�Ńp�����[�^���T���v�����O####
for(rp in 1:R){
  
  ##���͒P�ʂ̃g�s�b�N���T���v�����O
  #�Z�O�����g���Ƃ̖ޓx�𐄒�
  psi_log <- log(t(psi))
  LLi0 <- wx1_const + WX1 %*% psi_log 
  LLi_max <- apply(LLi0, 1, max)
  LLi <- exp(LLi0 - LLi_max)
  
  #�Z�O�����g�����m���̐���ƃZ�O�����g�̐���
  z1_rate <- matrix(0, nrow=a, ncol=k1)
  Zi1 <- matrix(0, nrow=a, ncol=k1)
  z1_vec <- rep(0, a)
  rf02 <- matrix(0, nrow=k1, ncol=k2) 
  
  for(j in 1:max_time){
    if(j==1){
      #�Z�O�����g�̊����m��
      LLs <- matrix(r0, nrow=length(index_t11), ncol=k1, byrow=T) * LLi[index_t11, ]   #�d�ݕt���ޓx
      z1_rate[index_t11, ] <- LLs / rowSums(LLs)   #�����m��
      
      #�������z���Z�O�����g�𐶐�
      Zi1[index_t11, ] <- rmnom(length(index_t11), 1, z1_rate[index_t11, ])
      z1_vec[index_t11] <- as.numeric(Zi1[index_t11, ] %*% 1:k1)
      
      #�������̃p�����[�^���X�V
      rf01 <- colSums(Zi1[index_t11, ])
      
    } else {
      
      #�Z�O�����g�̊����m��
      index <- index_t22[[j]]
      z1_vec[index_t21[[j]]]
      LLs <- r1[z1_vec[index_t21[[j]]], , drop=FALSE] * LLi[index, , drop=FALSE]   #�d�ݕt���ޓx
      z1_rate[index, ] <- LLs / rowSums(LLs)   #�����m��
      
      #�������z���Z�O�����g�𐶐�
      Zi1[index, ] <- rmnom(length(index), 1, z1_rate[index, ])
      z1_vec[index] <- as.numeric(Zi1[index, ] %*% 1:k1)
      
      #�������̃p�����[�^���X�V
      rf02 <- rf02 + t(Zi1[index_t21[[j]], , drop=FALSE]) %*% Zi1[index, , drop=FALSE]   #�}���R�t����
    }
  }
  
  #�f�B�N�������z���獬�������T���v�����O
  rf11 <- colSums(Zi1[index_t11, ]) + alpha01
  rf12 <- rf02 + alpha01
  r0 <- extraDistr::rdirichlet(1, rf11)
  r1 <- extraDistr::rdirichlet(k1, rf12)
  
  #�P�ꕪ�zpsi���T���v�����O
  df0 <- matrix(0, nrow=k1, ncol=v1)
  for(j in 1:k1){
    df0[j, ] <- colSums(WX1 * Zi1[, j])
  }
  df <- df0 + alpha01
  psi <- extraDistr::rdirichlet(k1, df)
  
  
  ##�P�ꂲ�ƂɃg�s�b�N�̐����ߒ����T���v�����O
  #�������L�̃g�s�b�N���z�̃p�����[�^�𐄒�
  word_par1 <- burden_fr(theta1, phi, wd, words2, k2)
  LLw1 <- rowSums(word_par1$Bur)
  
  #�Z�O�����g���L�̃g�s�b�N���z�̃p�����[�^�𐄒�
  word_par2 <- matrix(0, nrow=f2, ncol=k3)
  par2 <- theta2[rep(z1_vec, w2), ]
  for(j in 1:k3){
    word_par2[, j] <- par2[, j] * tau[j, wd]
  }
  LLw2 <- rowSums(word_par2)
  
  #�X�C�b�`���O�ϐ��𐶐�
  switching_rate <- r2*LLw1 / (r2*LLw1 + (1-r2)*LLw2)
  y <- rbinom(f2, 1, switching_rate)
  index_y <- which(y==1)
  
  #���������X�V
  par <- as.numeric(tapply(y, ID_d, sum))
  r02 <- rbeta(d, par+beta01, words2-par+beta01)
  r2 <- r02[ID_d]

  ##�P�ꂲ�ƂɃg�s�b�N���T���v�����O
  #�������L�̒P��g�s�b�N���T���v�����O
  n <- length(index_y)
  Zi2 <- matrix(0, nrow=f2, ncol=k2)
  Zi2[index_y, ] <- rmnom(n, 1, word_par1$Br[index_y, ])
  z2_vec <- as.numeric(Zi2 %*% 1:k2)
  
  #�Z�O�����g���L�̒P��g�s�b�N���T���v�����O
  word_rate2 <- word_par2 / rowSums(word_par2)   #�g�s�b�N�̊����m��
  Zi3 <- matrix(0, nrow=f2, ncol=k3)
  Zi3[-index_y, ] <- rmnom(f2-n, 1, word_rate2[-index_y, ])
  z3_vec <- as.numeric(Zi3 %*% 1:k3)

  
  ##�p�����[�^���T���v�����O
  #�������L�̃g�s�b�N���ztheta1���T���v�����O
  wsum0 <- matrix(0, nrow=d, ncol=k2)
  for(i in 1:d){
    wsum0[i, ] <- colSums(Zi2[doc_list[[i]], ])
  }
  wsum <- wsum0 + alpha02
  theta1 <- extraDistr::rdirichlet(d, wsum)
  
  #�Z�O�����g���L�̃g�s�b�N���ztheta2���T���v�����O
  dsum0 <- matrix(0, nrow=k1, ncol=k3)
  for(j in 1:k1){
    dsum0[j, ] <- colSums(Zi3 * rep(Zi1[, j], w2))
  }
  dsum <- dsum0 + alpha03
  theta2 <- extraDistr::rdirichlet(k1, dsum)

  
  #�P�ꕪ�zphi�����tau���T���v�����O
  vf0 <- matrix(0, nrow=k2, ncol=v2)
  tf0 <- matrix(0, nrow=k3, ncol=v2)
  for(j in 1:v2){
    vf0[, j] <- colSums(Zi2[word_list[[j]], ])
    tf0[, j] <- colSums(Zi3[word_list[[j]], ])
  }
  vf <- vf0 + beta01
  tf <- tf0 + beta02
  phi <- extraDistr::rdirichlet(k2, vf)
  tau <- extraDistr::rdirichlet(k3, tf)

  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^���i�[
  if(rp%%keep==0){
    #�T���v�����O���ʂ̊i�[
    mkeep <- rp/keep
    THETA1[, , mkeep] <- theta1
    THETA2[, , mkeep] <- theta2
    R0[mkeep, ] <- r0
    R1[, , mkeep] <- r1
    PSI[, , mkeep] <- psi
    PHI[, , mkeep] <- phi
    TAU[, , mkeep] <- tau
    
    #�g�s�b�N�����̓o�[���C�����Ԃ𒴂�����i�[����
    if(mkeep >= burnin & rp%%keep==0){
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
      SEG3 <- SEG3 + Zi3
    }
    
    #�T���v�����O���ʂ��m�F
    if(rp%%disp==0){
      LL1 <- sum(Zi1*LLi0)
      LL02 <- matrix(0, nrow=sum(y), ncol=k2)
      for(j in 1:k2){
        LL02[, j] <- phi[j, wd[index_y]]
      }
      LL2 <- sum(log(rowSums(LL02 * Zi2[index_y, ])))
      LL03 <- matrix(0, nrow=sum(1-y), ncol=k3)
      for(j in 1:k3){
        LL03[, j] <- tau[j, wd[-index_y]]
      }
      LL3 <- sum(log(rowSums(LL03 * Zi3[-index_y, ])))
      
      print(rp)
      print(round(c(mean(r02), mean(betat)), 3))
      print(c(LL1+LL2+LL3, LLst))
      print(round(cbind(theta1[1:7, ], thetat1[1:7, ]), 3))
      print(round(cbind(theta2[1:7, ], thetat2[1:7, ]), 3))
      print(round(cbind(phi[, 1:10], phit[, 1:10]), 3))
    }
  }
}


####�T���v�����O���ʂ̉����Ɨv��####
burnin <- 2000/keep   #�o�[���C������
RS <- R/keep

##�T���v�����O���ʂ̉���
#�����̃g�s�b�N���z�̃T���v�����O����
matplot(t(THETA1[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA1[100, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA1[1000, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA1[2000, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA2[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA2[3, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA2[5, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA2[7, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")

matplot(t(GAMMA[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(GAMMA[2, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(GAMMA[3, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(GAMMA[4, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")

#�P��̏o���m���̃T���v�����O����
matplot(t(PHI[1, 1:10, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N1�̒P��̏o�����̃T���v�����O����")
matplot(t(PHI[2, 51:60, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N2�̒P��̏o�����̃T���v�����O����")
matplot(t(PHI[3, 101:110, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N3�̒P��̏o�����̃T���v�����O����")
matplot(t(PHI[4, 151:160, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N4�̒P��̏o�����̃T���v�����O����")
matplot(t(PSI[1, 1:10, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N1�̒P��̏o�����̃T���v�����O����")
matplot(t(PSI[2, 51:60, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N2�̒P��̏o�����̃T���v�����O����")
matplot(t(PSI[3, 101:110, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N3�̒P��̏o�����̃T���v�����O����")
matplot(t(PSI[4, 151:160, ]), type="l", ylab="�p�����[�^", main="�g�s�b�N4�̒P��̏o�����̃T���v�����O����")

#��ʌ�̏o���m���̃T���v�����O����
matplot(GAMMA[, 286:295], type="l", ylab="�p�����[�^", main="�P��̏o�����̃T���v�����O����")
matplot(GAMMA[, 296:305], type="l", ylab="�p�����[�^", main="�P��̏o�����̃T���v�����O����")
matplot(GAMMA[, 306:315], type="l", ylab="�p�����[�^", main="�P��̏o�����̃T���v�����O����")


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