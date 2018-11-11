#####����LDA���f��#####
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
k1 <- 10   #�P�ꂲ�Ƃ̃g�s�b�N��
k2 <- 7   #���͂̃Z�O�����g��
k3 <- 10   #���͂��Ƃ̃g�s�b�N��
d <- 2000   #������
v <- 350   #��b��
s <- rpois(d, 15)   #���͐�
s[s < 5] <- ceiling(runif(sum(s < 5), 5, 10))
a <- sum(s)   #�����͐�
w <- rpois(a, 13)   #���͂�����̒P�ꐔ
w[w < 5] <- ceiling(runif(sum(w < 5), 5, 10))
f <- sum(w)   #���P�ꐔ

#����ID�̐ݒ�
u_id <- rep(1:d, s)
t_id <- c()
for(i in 1:d){t_id <- c(t_id, 1:s[i])}
words <- as.numeric(tapply(w, u_id, sum))


##�p�����[�^��ݒ�
#�f�B���N�����z�̃p�����[�^
alpha01 <- rep(4.0, k2)
alpha02 <- rep(0.15, k3)
alpha03 <- rep(0.25, k1)
alpha11 <- rep(0.4, v)
alpha12 <- rep(0.1, v)

#�f�B���N�����z���p�����[�^�𐶐�
omegat <- omega <- extraDistr::rdirichlet(1, alpha01)   #�Z�O�����g�����m��
gammat <- gamma <- extraDistr::rdirichlet(k2, alpha02)   #�����Z�O�����g�P�ʂ̃g�s�b�N���z
thetat <- theta <- extraDistr::rdirichlet(d, alpha03)   #�����P�ʂ̃g�s�b�N���z
phit <- phi <- extraDistr::rdirichlet(k1, alpha11)
psit <- psi <- extraDistr::rdirichlet(k3, alpha12)
betat <- beta <- 0.45

##���͂��ƂɒP��𐶐�����
WX <- matrix(0, nrow=a, ncol=v)
y_list <- list()
Z1_list <- list()
Z2_list <- list()

for(i in 1:a){

  ##���͂��ƂɃg�s�b�N���z�𐶐�
  y <- rbinom(1, 1, beta)
  
  ##���͂��ƂɃg�s�b�N�𐶐�
  id <- u_id[i]
  if(y==1){
    z2 <- rmnom(w[i], 1, theta[id, ])
    z2_vec <- as.numeric(z2 %*% 1:k1)
    z1 <- rep(0, k2)
  } else {
    z1 <- rmnom(1, 1, omega)
    z1_vec <- as.numeric(z1 %*% 1:k2)
    z2 <- rmnom(w[i], 1, gamma[z1_vec, ])
    z2_vec <- as.numeric(z2 %*% 1:k3)
  }
  
  #�g�s�b�N���z�Ɋ�Â��P��𐶐�
  if(y==1){
    wn <- rmnom(w[i], 1, phi[z2_vec, ])
  } else {
    wn <- rmnom(w[i], 1, psi[z2_vec, ])
  }
  WX[i, ] <- colSums(wn)
  
  #�p�����[�^���i�[
  Z1_list[[i]] <- z1
  Z2_list[[i]] <- z2_vec
  y_list[[i]] <- y
}

#���X�g�`����ϊ�
Z1 <- do.call(rbind, Z1_list)
Z2 <- unlist(Z2_list)
Y <- unlist(y_list)
z2_freq <- as.numeric(table(Z2[Y==0]))


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
  ID_list[[i]] <- rep(u_id[i], w[i])
  td1_list[[i]] <- rep(i, w[i])
  td2_list[[i]] <- rep(t_id[i], w[i])
  
  #�P��ID���L�^
  num1 <- WX[i, ] * 1:v
  num2 <- which(num1 > 0)
  W1 <- WX[i, (WX[i, ] > 0)]
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
for(i in 1:v){word_list[[i]] <- which(wd==i)}


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


##�A���S���Y���̐ݒ�
R <- 5000
keep <- 2  
iter <- 0
burnin <- 1000/keep
disp <- 10

##���O���z�̐ݒ�
#�n�C�p�[�p�����[�^�̎��O���z
alpha01 <- 1 
alpha02 <- 1
alpha03 <- 25
beta01 <- 0.5


##�p�����[�^�̏����l
#tfidf�ŏ����l��ݒ�
tf0 <- colMeans(WX)*10
idf1 <- log(nrow(WX)/colSums(WX > 0))
idf2 <- log(nrow(WX)/colSums(WX==0))

#�����l�ݒ�̂��߂Ƀg�s�b�N���f���𐄒�
#�P��g�s�b�N�P�ʂ̃p�����[�^�̏����l
theta <- extraDistr::rdirichlet(d, rep(10, k1))   #�����P�ʂ̃g�s�b�N�̏����l
phi <- extraDistr::rdirichlet(k1, tf0)   #�����P�ʂ̏o���m���̏����l
gamma <- extraDistr::rdirichlet(k2, rep(10, k3))   #���̓Z�O�����g�̃g�s�b�N�̏����l
psi <- extraDistr::rdirichlet(k3, rep(0.2, v))   #���̓Z�O�����g�̏o���m���̏����l
r <- c(0.5, rep(0.5/k2, k2))

##�p�����[�^�̊i�[�p�z��
THETA <- array(0, dim=c(d, k1, R/keep))
PHI <- array(0, dim=c(k1, v, R/keep))
GAMMA <- array(0, dim=c(k2, k3, R/keep))
PSI <- array(0, dim=c(k3, v, R/keep))
OMEGA <- matrix(0, nrow=R/keep, ncol=k2+1)
SEG1 <- matrix(0, nrow=a, ncol=k2+1)
SEG2 <- matrix(0, nrow=f, ncol=k1)
SEG3 <- matrix(0, nrow=f, ncol=k3)
storage.mode(SEG1) <- "integer"
storage.mode(SEG2) <- "integer"
storage.mode(SEG3) <- "integer"

##MCMC����p�z��
wsum0 <- matrix(0, nrow=d, ncol=k1)
vf0 <- matrix(0, nrow=k1, ncol=v)
dsum0 <- matrix(0, nrow=d, ncol=k2)
sf0 <- matrix(0, nrow=k2, ncol=v)



####�M�u�X�T���v�����O�Ńp�����[�^���T���v�����O####
for(rp in 1:R){
  
  ##�P�ꂲ�Ƃ̃g�s�b�N�̖ޓx�𐄒�
  #�����P�ʂł̃g�s�b�N�ޓx
  word_par1 <- burden_fr(theta, phi, wd, words, k1)

  #���͒P�ʂ̃g�s�b�N�ޓx
  LLsums <- matrix(0, nrow=f, ncol=k2)
  for(j in 1:k3){
    LLsums <- LLsums + psi[j, wd] * matrix(gamma[, j], nrow=f, ncol=k2, byrow=T)
  }
  
  ##���͒P�ʂ̃g�s�b�N�𐶐�
  #���͒P�ʂ̖ޓx�ƃg�s�b�N�����m��
  LLind <- cbind(rowSums(word_par1$Bur), LLsums)
  LLho <- matrix(0, nrow=a, ncol=k2+1)
  for(j in 1:a){
    LLho[j, ] <- r * colProds(LLind[sent_list[[j]], ])
  }

  #�������z���̕��̓g�s�b�N�𐶐�
  switch_rate <- LLho / rowSums(LLho)   #���̓g�s�b�N�̊����m��
  Zi1 <- rmnom(a, 1, switch_rate)
  zi1_vec <- as.numeric(Zi1 %*% 1:(k2+1))
  index_z1 <- which(rep(Zi1[, 1], w)==1)
  
  #���������X�V
  r0 <- colSums(Zi1)
  r <- as.numeric(extraDistr::rdirichlet(1, r0 + alpha03))
  
  
  ##�������z���P��g�s�b�N�𐶐�
  #�����P�ʂł̒P��g�s�b�N�̐���
  Zi2 <- matrix(0, nrow=f, ncol=k1)
  Zi2[index_z1, ] <- rmnom(length(index_z1), 1, word_par1$Br[index_z1, ])
  zi2_vec <- as.numeric(Zi2 %*% 1:k1)

  #���̓Z�O�����g�P�ʂł̒P��g�s�b�N�̖ޓx
  index_seg <- rep(zi1_vec-1, w)[-index_z1]
  word_vec <- wd[-index_z1]
  n <- length(index_seg)
  word_par2 <- matrix(0, nrow=n, ncol=k3)
  
  for(j in 1:k3){
    word_par2[, j] <- psi[j, word_vec] * gamma[index_seg, j]
  }
  
  #�������z���P��g�s�b�N�𐶐�
  word_rate2 <- word_par2 / rowSums(word_par2)   #���݃g�s�b�N�̊����m��
  Zi3 <- matrix(0, nrow=f, ncol=k3)
  Zi3[-index_z1, ] <- rmnom(n, 1, word_rate2)
  zi3_vec <- as.numeric(Zi3 %*% 1:k3)
  Zi3_hat <- Zi3[-index_z1, ]


  ##�p�����[�^���T���v�����O
  #�����P�ʂ̃g�s�b�N���ztheta���T���v�����O
  wsum0 <- matrix(0, nrow=d, ncol=k1)
  for(i in 1:d){
    wsum0[i, ] <- colSums(Zi2[doc_list[[i]], ])
  }
  wsum <- wsum0 + alpha01
  theta <- extraDistr::rdirichlet(d, wsum)
  
  #���͒P�ʂ̃g�s�b�N���zgamma���T���v�����O
  vsum0 <- matrix(0, nrow=k2, ncol=k3)
  for(j in 1:k2){
    vsum0[j, ] <- colSums(Zi3_hat[index_seg==j, ])
  }
  vsum <- vsum0 + alpha02 
  gamma <- extraDistr::rdirichlet(k2, vsum)
  
  #�P�ꕪ�zphi�����psi���T���v�����O
  vf0 <- matrix(0, nrow=k1, ncol=v)
  df0 <- matrix(0, nrow=k3, ncol=v)
  for(j in 1:v){
    vf0[, j] <- colSums(Zi2[word_list[[j]], ])
    df0[, j] <- colSums(Zi3[word_list[[j]], ])
  }
  
  vf <- vf0 + alpha01
  df <- df0 + alpha02
  phi <- extraDistr::rdirichlet(k1, vf)
  psi <- extraDistr::rdirichlet(k3, df)
  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^���i�[
  if(rp%%keep==0){
    #�T���v�����O���ʂ̊i�[
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    GAMMA[, , mkeep] <- gamma
    PSI[, , mkeep] <- psi
    OMEGA[mkeep, ] <- r
    
    #�g�s�b�N�����̓o�[���C�����Ԃ𒴂�����i�[����
    if(mkeep >= burnin & rp%%keep==0){
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
      SEG3 <- SEG3 + Zi3
    }
    
    #�T���v�����O���ʂ��m�F
    if(rp%%disp==0){
      print(rp)
      print(rbind(df0=rowSums(df0), z2_freq))
      print(round(rbind(r, r0=c(betat, (1-betat)*omegat)), 3))
      print(round(cbind(theta[1:7, ], thetat[1:7, ]), 3))
      print(round(cbind(psi[, 1:10], psit[, 1:10]), 3))
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