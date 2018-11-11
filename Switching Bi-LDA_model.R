#####Switching Bi-LDA���f��#####
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

#set.seed(21437)

####�f�[�^�̔���####
##�f�[�^�̐ݒ�
hh0 <- 1000
item0 <- 75

##ID�ƃ��r���[�����𔭐�
#ID�����ݒ�
u.id0 <- rep(1:hh0, rep(item0, hh0))
i.id0 <- rep(1:item0, hh0)

#���r���[�����𔭐�
buy_hist <- rep(0, hh0*item0)
for(i in 1:item0){
  p <- runif(1, 0.05, 0.3)
  buy_hist[i.id0==i] <- rbinom(hh0, 1, p)
}

#���r���[��������ID���Đݒ�
index <- which(buy_hist==1)
u.id <- u.id0[index]
u.freq <- plyr::count(u.id)[, 2]
i.id <- i.id0[index]
i.freq <- plyr::count(i.id)[, 2]
ID <- data.frame(no=1:length(u.id), id=u.id, item=i.id)

#�f�[�^�̍Đݒ�
k1 <- 8   #���[�U�[�g�s�b�N��
k2 <- 10   #�A�C�e���g�s�b�N��
hh <- length(unique(u.id))   #���[�U�[��
item <- length(unique(i.id))   #�A�C�e����
d <- length(u.id)   #������
v <- 300   #��b��
v1 <- 175   #���[�U�[�]���Ɋ֌W�̂����b��
v2 <- v-v1   #�A�C�e���]���Ɋ֌W�̂����b��

#1����������̒P�ꐔ
w <- rpois(d, rgamma(d, 50, 1.0))   #1����������̒P�ꐔ
w[w < 25] <- ceiling(runif(sum(w < 25), 25, 35))
f <- sum(w)   #���P�ꐔ


####bag of word�`���̕����s��𔭐�####
##�p�����[�^�̐ݒ�
#�f�B���N�����O���z�̃p�����[�^��ݒ�
alpha01 <- rep(0.25, k1)   #���[�U�[�g�s�b�N�̃f�B���N�����O���z�̃p�����[�^
alpha02 <- rep(0.3, k2)   #�A�C�e���g�s�b�N�̃f�B���N�����O���z�̃p�����[�^
alpha11 <- c(rep(0.4, v1), rep(0.0075, v2))   #���[�U�[�̒P�ꕪ�z�̃f�B���N�����O���z�̃p�����[�^
alpha12 <- c(rep(0.0075, v1), rep(0.45, v2))   #�A�C�e���̒P�ꕪ�z�̃f�B���N�����O���z�̃p�����[�^

#�f�B���N�����O���z����p�����[�^�𐶐�
thetat <- theta <- extraDistr::rdirichlet(hh, alpha01)   #���[�U�[�g�s�b�N�̐���
gammat <- gamma <- extraDistr::rdirichlet(item, alpha02)   #�A�C�e���g�s�b�N�̐���
phit <- phi <- extraDistr::rdirichlet(k1, alpha11)   #���[�U�[�̒P�ꕪ�z�𐶐�
lambdat <- lambda <- extraDistr::rdirichlet(k2, alpha12)   #�A�C�e���̒P�ꕪ�z�𐶐�

#���[�U�[���A�C�e���̃X�C�b�`���O�ϐ��̃p�����[�^
omega <- rbeta(hh, 8, 10)


##�������z����g�s�b�N����ђP��f�[�^�𐶐�
WX <- matrix(0, nrow=d, ncol=v)
Z01_list <- list()
Z02_list <- list()
y0_list <- list()
u.id_list <- list()
i.id_list <- list()
index_word1 <- 1:v1
index_word2 <- (v1+1):v

##���r���[���ƂɒP��𐶐�����
for(i in 1:hh){
  print(i)
  
  for(j in 1:u.freq[i]){
    word <- w[u.id==i][j]   #��������P�ꐔ
    index_row <- which(u.id==i)[j]
    
    #���[�U�[�g�s�b�N�𐶐�
    z1 <- rmnom(word, 1, theta[i, ])
    z1_vec <- z1 %*% 1:k1
    
    #�A�C�e���g�s�b�N�𐶐�
    z2 <- rmnom(word, 1, gamma[i.id[index_row], ])
    z2_vec <- z2 %*% 1:k2
    
    #�X�C�b�`���O�ϐ��𐶐�
    y <- rbinom(word, 1, omega[i])
    index_y <- which(y==1)
    
    #�p�����[�^����P��𐶐�
    user_word <- colSums(rmnom(length(index_y), 1, phi[z1_vec[index_y], ]))
    item_word <- colSums(rmnom(word-length(index_y), 1, lambda[z2_vec[-index_y], ]))
    
    #�f�[�^���i�[
    WX[index_row, ] <- user_word + item_word
    Z01_list[[index_row]] <- z1
    Z02_list[[index_row]] <- z2
    y0_list[[index_row]] <- y
    u.id_list[[index_row]] <- rep(i, word)
    i.id_list[[index_row]] <- rep(i.id[index_row], word)
  }
}

##���X�g�`�����f�[�^�`���ɕϊ�
Z02 <- do.call(rbind, Z02_list)
Z01 <- do.call(rbind, Z01_list)
y0 <- unlist(y0_list)
u.id_vec <- unlist(u.id_list)
i.id_vec <- unlist(i.id_list)
storage.mode(Z02) <- "integer"
storage.mode(Z01) <- "integer"
storage.mode(WX) <- "integer"


####�g�s�b�N���f������̂��߂̃f�[�^�̏���####
##�f�[�^����p��ID���쐬
user_list <- list()
item_list <- list()
wd_list <- list()

#ID���Ƃ�tweet_id����ђP��id���쐬
for(i in 1:hh){
  print(i)
  
  for(j in 1:u.freq[i]){
    index_row <- which(u.id==i)[j]
    
    #���[�U�[ID�ƃA�C�e��ID���L�^
    user_list[[index_row]] <- rep(i, w[index_row])
    item_list[[index_row]] <- rep(i.id[index_row], w[index_row])
    
    #�P��ID���L�^
    num1 <- WX[index_row, ] * 1:v
    num2 <- which(num1 > 0)
    W1 <- WX[index_row, (WX[index_row, ] > 0)]
    wd_list[[index_row]] <- rep(num2, W1)
  }
}

#���X�g���x�N�g���ɕϊ�
user_id <- unlist(user_list)
item_id <- unlist(item_list)
wd <- unlist(wd_list)

##�C���f�b�N�X���쐬
user_list <- list()
item_list <- list()
word_list <- list()
for(i in 1:length(unique(user_id))){user_list[[i]] <- which(user_id==i)}
for(i in 1:length(unique(item_id))){item_list[[i]] <- which(item_id==i)}
for(i in 1:length(unique(wd))){word_list[[i]] <- which(wd==i)}


####�}���R�t�A�������e�J�����@��Switching Bi-LDA���f���𐄒�####
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
R <- 10000
keep <- 2
iter <- 0
burnin <- 1000/keep

##���O���z�̐ݒ�
alpha01 <- 1
alpha02 <- 1
beta01 <- 0.5
gamma01 <- 0.5

##�p�����[�^�̏����l
#�g�s�b�N���z�̏����l
theta <- extraDistr::rdirichlet(hh, rep(5, k1))   #���[�U�[�g�s�b�N�̏����l
gamma <- extraDistr::rdirichlet(item, rep(5, k2))   #�A�C�e���g�s�b�N�̏����l

#���[�U�[�̒P�ꕪ�z�̏����l
u_word <- as.matrix(data.frame(id=u.id, WX) %>%
                      dplyr::group_by(id) %>%
                      dplyr::summarise_all(funs(sum)))[, 2:(v+1)]
phi <- extraDistr::rdirichlet(k1, apply(u_word, 2, sd))

#�A�C�e���̒P�ꕪ�z�̏����l
i_word <- as.matrix(data.frame(id=i.id, WX) %>%
                      dplyr::group_by(id) %>%
                      dplyr::summarise_all(funs(sum)))[, 2:(v+1)]
lambda <- extraDistr::rdirichlet(k2, apply(i_word, 2, sd)/10)

#���[�U�[�ƃA�C�e���̍������̏����l
y <- rbinom(f, 1, 0.5)
r <- matrix(c(0.5, 0.5), nrow=hh, ncol=2)


##�p�����[�^�̕ۑ��p�z��
THETA <- array(0, dim=c(hh, k1, R/keep))
GAMMA <- array(0, dim=c(item, k2, R/keep))
PHI <- array(0, dim=c(k1, v, R/keep))
LAMBDA <- array(0, dim=c(k2, v, R/keep))
SEG1 <- matrix(0, nrow=f, ncol=k1)
SEG2 <- matrix(0, nrow=f, ncol=k2)
Y <- rep(0, f)
storage.mode(SEG1) <- "integer"
storage.mode(SEG2) <- "integer"

##MCMC����p�z��
phi_d <- matrix(0, nrow=f, ncol=k1)
lambda_d <- matrix(0, nrow=f, ncol=k2)
Bur1 <- matrix(0, nrow=f, ncol=k1)
Bur2 <- matrix(0, nrow=f, ncol=k2)
LH1 <- rep(0, f)
usum0 <- matrix(0, nrow=hh, ncol=k1)
isum0 <- matrix(0, nrow=item, ncol=k2)
uvf0 <- matrix(0, nrow=k1, ncol=v)
ivf0 <- matrix(0, nrow=k2, ncol=v)
vec1 <- 1/1:k1
vec2 <- 1/1:k2


####�M�u�X�T���v�����O�Ńp�����[�^���T���v�����O####
for(rp in 1:R){
  
  ##�P�ꂲ�ƂɃ��[�U�[�g�s�b�N�ƃA�C�e���g�s�b�N�𐶐�
  #���r���[�P�ʂɊg�債���p�����[�^�s��
  theta_d <- theta[u.id, ]
  gamma_d <- gamma[i.id, ]
  
  #���[�U�[�P�ʂ̖ޓx�ƃg�s�b�N�����m���𐄒�
  for(j in 1:k1){
    phi_d[, j] <- phi[j, wd]
    Bi1 <- rep(theta_d[, j], w) * phi_d[, j]
    Bur1[, j] <- Bi1
  }
  user_rate <- Bur1 / rowSums(Bur1)
  
  #�A�C�e���P�ʂ̖ޓx�ƃg�s�b�N�����m���𐄒�
  for(j in 1:k2){
    lambda_d[, j] <- lambda[j, wd]
    Bi2 <- rep(gamma_d[, j], w) * lambda_d[, j]
    Bur2[, j] <- Bi2
  }
  item_rate <- Bur2 / rowSums(Bur2)
  
  ##�X�C�b�`���O�ϐ��𐶐�
  #�X�C�b�`���O�ϐ��̃p�����[�^��ݒ�
  rd <- r[user_id, ]
  #LLz1 <- rd[, 1] * rowSums(user_rate*phi_d)
  #LLz2 <- rd[, 2] * rowSums(item_rate*lambda_d)
  LLz1 <- rd[, 1] * rowSums(Bur1)
  LLz2 <- rd[, 2] * rowSums(Bur2)
  switching_rate <- LLz1 / (LLz1+LLz2)
  
  #�x���k�[�C���z���X�C�b�`���O�ϐ��𐶐�
  y <- rbinom(f, 1, switching_rate)
  r1 <- tapply(y, user_id, mean)
  r <- cbind(r1, 1-r1) 
  
  ##���������X�C�b�`���O�ϐ��Ɋ�Â��g�s�b�N�𐶐�
  index_y <- which(y==1)
  Zi1 <- matrix(0, nrow=f, ncol=k1)
  Zi2 <- matrix(0, nrow=f, ncol=k2)
  z1_vec <- rep(0, f)
  z2_vec <- rep(0, f)
  
  #�������z��胆�[�U�[�g�s�b�N�𐶐�
  n <- length(index_y)
  rand1 <- matrix(runif(n), nrow=n, ncol=k1)
  user_cumsums <- rowCumsums(user_rate[index_y, ])
  z1 <- ((k1+1) - (user_cumsums > rand1) %*% rep(1, k1)) %*% vec1   #�g�s�b�N���T���v�����O
  z1[z1!=1] <- 0
  Zi1[index_y, ] <- z1
  z1_vec[index_y] <- z1 %*% 1:k1
  
  #�������z���A�C�e���g�s�b�N�𐶐�
  n <- f-length(index_y)
  rand2 <- matrix(runif(n), nrow=n, ncol=k2)
  item_cumsums <- rowCumsums(item_rate[-index_y, ])
  z2 <- ((k2+1) - (item_cumsums > rand2) %*% rep(1, k2)) %*% vec2   #�g�s�b�N���T���v�����O
  z2[z2!=1] <- 0
  Zi2[-index_y, ] <- z2
  z2_vec[-index_y] <- z2 %*% 1:k2
  
  
  ##���[�U�[�g�s�b�N�̃p�����[�^���T���v�����O
  for(i in 1:hh){
    usum0[i, ] <- colSums(Zi1[user_list[[i]], ])
  }
  usum <- usum0 + alpha01
  theta <- extraDistr::rdirichlet(hh, usum)
  
  ##�A�C�e���g�s�b�N�̃p�����[�^���T���v�����O
  for(i in 1:item){
    isum0[i, ] <- colSums(Zi2[item_list[[i]], ])
  }
  isum <- isum0 + alpha02
  gamma <- extraDistr::rdirichlet(item, isum)
  
  
  ##���[�U�[�ƃA�C�e�����x���̒P�ꕪ�z���T���v�����O
  for(j in 1:v){
    uvf0[, j] <- colSums(Zi1[word_list[[j]], ])
    ivf0[, j] <- colSums(Zi2[word_list[[j]], ])
  }
  uvf <- uvf0 + beta01
  ivf <- ivf0 + gamma01
  phi <- extraDistr::rdirichlet(k1, uvf)
  lambda <- extraDistr::rdirichlet(k2, ivf)
  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^���i�[
  if(rp%%keep==0){
    #�T���v�����O���ʂ̊i�[
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    GAMMA[, , mkeep] <- gamma
    PHI[, , mkeep] <- phi
    LAMBDA[, , mkeep] <- lambda
    
    #�g�s�b�N�����̓o�[���C�����Ԃ𒴂�����i�[����
    if(mkeep >= burnin & rp%%keep==0){
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
      Y <- Y + y
    }
    
    #�T���v�����O���ʂ��m�F
    print(rp)
    print(round(cbind(theta[1:10, ], thetat[1:10, ]), 3))
    #print(round(cbind(gamma[1:10, ], gammat[1:10, ]), 3))
    print(round(cbind(phi[, 171:180], phit[, 171:180]), 3))
    #print(round(cbind(lambda[, 171:180], lambdat[, 171:180]), 3))
  }
}

####�T���v�����O���ʂ̉����Ɨv��####
burnin <- 1000/keep
RS <- R/keep

##�T���v�����O���ʂ̃v���b�g
matplot(t(THETA[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA[250, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA[1000, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(GAMMA[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(GAMMA[25, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(GAMMA[50, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI[, 1, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI[, 175, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI[, 176, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(LAMBDA[, 175, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(LAMBDA[, 176, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(LAMBDA[, 200, ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")

##�T���v�����O���ʂ̗v��
#�T���v�����O���ʂ̎��㕽��
round(cbind(apply(THETA[, , burnin:RS], c(1, 2), mean), thetat), 3)   #���[�U�[�̃g�s�b�N�����̎��㕽��
round(cbind(apply(GAMMA[, , burnin:RS], c(1, 2), mean), gammat), 3)   #�A�C�e���̃g�s�b�N�����̎��㕽��
round(cbind(t(apply(PHI[, , burnin:RS], c(1, 2), mean)), t(phit)), 3)   #���[�U�[�̒P�ꕪ�z�̎��㕽��
round(cbind(t(apply(LAMBDA[, , burnin:RS], c(1, 2), mean)), t(lambdat)), 3)   #�A�C�e���̒P�ꕪ�z�̎��㕽��

#�T���v�����O���ʂ̎���M�p���
round(apply(THETA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025)), 3)
round(apply(THETA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975)), 3)
round(apply(GAMMA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025)), 3)
round(apply(GAMMA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975)), 3)
round(t(apply(PHI[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025))), 3)
round(t(apply(PHI[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975))), 3)
round(t(apply(LAMBDA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025))), 3)
round(t(apply(LAMBDA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975))), 3)

##�T���v�����O���ꂽ���ݕϐ��̗v��
n <- length(burnin:RS)
round(cbind(Y / n, wd), 3)
round(cbind(wd, y=Y/n, SEG1/rowSums(SEG1)), 3)
round(cbind(wd, y=Y/n, SEG2/rowSums(SEG2)), 3)
