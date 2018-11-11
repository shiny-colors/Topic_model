#####Switching Multinom LDA model#####
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
#set.seed(2506787)

####�f�[�^�̔���####
##�f�[�^�̐ݒ�
r <- 5   #�]���X�R�A��
s <- 3   #�ɐ��l��
a <- 3   #����
k11 <- 5   #���[�U�[�̕]���X�R�A�̃g�s�b�N��
k12 <- 5   #�A�C�e���̕]���X�R�A�̃g�s�b�N��
K1 <- matrix(1:(k11*k12), nrow=k11, ncol=k12, byrow=T)   #�g�s�b�N�̔z��
k21 <- 15   #���[�U�[�̃e�L�X�g�̃g�s�b�N��
k22 <- 15   #�A�C�e���̃e�L�X�g�̃g�s�b�N��
hh <- 3000   #���r���A�[��
item <- 1500   #�A�C�e����
v1 <- 300   #�]���X�R�A�̌�b��
v2 <- 400   #���[�U�[�g�s�b�N�̌�b��
v3 <- 400   #�A�C�e���g�s�b�N�̌�b��
v <- v1 + v2 + v3   #����b��
spl <- matrix(1:v1, nrow=s, ncol=v1/s, byrow=T)
v1_index <- 1:v1
v2_index <- (v1+1):v2
v3_index <- (v2+1):v

##ID�ƌ����x�N�g���̍쐬
#ID�����ݒ�
user_id0 <- rep(1:hh, rep(item, hh))
item_id0 <- rep(1:item, hh)

#�����x�N�g�����쐬
repeat { 
  m_vec <- rep(0, hh*item)
  for(i in 1:item){
    prob <- runif(1, 0.005, 0.07)
    m_vec[item_id0==i] <- rbinom(hh, 1, prob)
  }
  m_index <- which(m_vec==1)
  
  #���S��ID��ݒ�
  user_id <- user_id0[m_index]
  item_id <- item_id0[m_index]
  d <- length(user_id)   #�����r���[��
  
  #���ׂẴp�^�[��������������break
  if(length(unique(user_id))==hh & length(unique(item_id))==item) break
}

#�P�ꐔ��ݒ�
w <- rpois(d, rgamma(d, 25, 0.5))   #����������̒P�ꐔ
f <- sum(w)   #���P�ꐔ
n_user <- plyr::count(user_id)$freq
n_item <- plyr::count(item_id)$freq

#�P��ID��ݒ�
u_id <- rep(user_id, w)
i_id <- rep(item_id, w)
d_id <- rep(1:d, w)


#�C���f�b�N�X��ݒ�
user_index <- list()
item_index <- list()
for(i in 1:hh){
  user_index[[i]] <- which(user_id==i)
}
for(j in 1:item){
  item_index[[j]] <- which(item_id==j)
}

##�p�����[�^�̐ݒ�
#�g�s�b�N���z�̎��O���z�̐ݒ�
alpha11 <- rep(0.2, k11); alpha12 <- rep(0.2, k12)
alpha21 <- rep(0.15, k21); alpha22 <- rep(0.15, k22)
alpha3 <- c(0.1, 0.225, 0.3, 0.25, 0.125) * r
beta1 <- c(1.6, 4.8, 5.6)   #�X�C�b�`���O�ϐ��̎��O���z

#�]���X�R�A�̒P�ꕪ�z�̎��O���z�̐ݒ�
alpha41 <- c(rep(0.5, v1/s), rep(0.025, v1/s), rep(0.0025, v1/s), rep(0.001, v2), rep(0.001, v3))
alpha42 <- c(rep(0.3, v1/s), rep(0.1, v1/s), rep(0.025, v1/s), rep(0.001, v2), rep(0.001, v3))
alpha43 <- c(rep(0.2, v1/s), rep(1.0, v1/s), rep(0.2, v1/s), rep(0.001, v2), rep(0.001, v3))
alpha44 <- c(rep(0.025, v1/s), rep(0.1, v1/s), rep(0.3, v1/s), rep(0.001, v2), rep(0.001, v3))
alpha45 <- c(rep(0.0025, v1/s), rep(0.025, v1/s), rep(0.5, v1/s), rep(0.001, v2), rep(0.001, v3))
alpha4 <- rbind(alpha41, alpha42, alpha43, alpha44, alpha45)

#���[�U�[�ƃA�C�e���̒P�ꕪ�z�̎��O���z�̐ݒ�
alpha51 <- c(rep(0.0001, v1/s), rep(0.0001, v1/s), rep(0.0001, v1/s), rep(0.05, v2), rep(0.0001, v3))
alpha52 <- c(rep(0.0001, v1/s), rep(0.0001, v1/s), rep(0.0001, v1/s), rep(0.001, v2), rep(0.05, v3))


##���ׂĂ̒P�ꂪ�o������܂Ńf�[�^�̐����𑱂���
rp <- 0 
repeat {
  rp <- rp + 1
  print(rp)
  
  #�g�s�b�N���z�̃p�����[�^�𐶐�
  theta11 <- thetat11 <- extraDistr::rdirichlet(hh, alpha11)
  theta12 <- thetat12 <- extraDistr::rdirichlet(item, alpha12)
  theta21 <- thetat21 <- extraDistr::rdirichlet(hh, alpha21)
  theta22 <- thetat22 <- extraDistr::rdirichlet(item, alpha22)
  eta <- etat <- extraDistr::rdirichlet(k11*k12, alpha3)
  lambda <- lambdat <- extraDistr::rdirichlet(hh, beta1)   #�X�C�b�`���O�ϐ��̃p�����[�^
  
  #�P�ꕪ�z�̃p�����[�^�𐶐�
  omega <- omegat <- extraDistr::rdirichlet(r, alpha4)
  phi <- phit <- extraDistr::rdirichlet(k21, alpha51)
  gamma <- gammat <- extraDistr::rdirichlet(k22, alpha52)
  
  #�o���m�����Ⴂ�P������ւ���
  index <- which(colMaxs(phi[, alpha51==max(alpha51)]) < (k21*k21)/f)
  for(j in 1:length(index)){
    phi[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, rep(1, k21))) %*% 1:k21), index[j]] <- (k21*k21)/f
  }
  index <- which(colMaxs(gamma[, alpha52==max(alpha52)]) < (k22*k22)/f)
  for(j in 1:length(index)){
    gamma[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, rep(1, k22))) %*% 1:k22), index[j]] <- (k22*k22)/f
  }
  
  ##���f���Ɋ�Â��f�[�^�𐶐�
  WX <- matrix(0, nrow=d, ncol=v)
  y <- rep(0, d)
  U1 <- matrix(0, nrow=d, ncol=k11)
  U2 <- matrix(0, nrow=d, ncol=k12)
  Z1_list <- Z21_list <- Z22_list <- wd_list <- list()
  
  for(i in 1:d){
    #���[�U�[�ƃA�C�e���𒊏o
    u_index <- user_id[i]
    i_index <- item_id[i]
    
    #�]���X�R�A�̃g�s�b�N�𐶐�
    u1 <- as.numeric(rmnom(1, 1, theta11[u_index, ]))
    u2 <- as.numeric(rmnom(1, 1, theta12[i_index, ]))
    
    #�]���X�R�A�̃g�s�b�N����X�R�A�𐶐�
    y[i] <- as.numeric(rmnom(1, 1, eta[K1[which.max(u1), which.max(u2)], ]) %*% 1:r)
    
    #�������z����X�C�b�`���O�ϐ��𐶐�
    z1 <- rmnom(w[i], 1, lambda[u_index, ])
    z1_vec <- as.numeric(z1 %*% 1:a)
    index_z11 <- which(z1[, 1]==1)
    
    #���[�U�[�g�s�b�N�𐶐�
    z21 <- matrix(0, nrow=w[i], ncol=k21)
    index_z21 <- which(z1[, 2]==1)
    if(sum(z1[, 2]) > 0){
      z21[index_z21, ] <- rmnom(sum(z1[, 2]), 1, theta21[u_index, ])
    }
    z21_vec <- as.numeric(z21 %*% 1:k21)
    
    #�A�C�e���g�s�b�N�𐶐�
    z22 <- matrix(0, nrow=w[i], ncol=k22)
    index_z22 <- which(z1[, 3]==1)
    if(sum(z1[, 3]) > 0){
      z22[index_z22, ] <- rmnom(sum(z1[, 3]), 1, theta22[i_index, ])
    }
    z22_vec <- as.numeric(z22 %*% 1:k22)
    
    #�g�s�b�N����P��𐶐�
    words <- matrix(0, nrow=w[i], ncol=v)
    if(sum(z1[, 1]) > 0){
      words[index_z11, ] <- rmnom(sum(z1[, 1]), 1, omega[y[i], ])
    }
    if(sum(z1[, 2]) > 0){
      words[index_z21, ] <- rmnom(sum(z1[, 2]), 1, phi[z21_vec[index_z21], ])
    }
    if(sum(z1[, 3]) > 0){
      words[index_z22, ] <- rmnom(sum(z1[, 3]), 1, gamma[z22_vec[index_z22], ])
    }
    word_vec <- as.numeric(words %*% 1:v)
    WX[i, ] <- colSums(words)
    
    #�f�[�^���i�[
    wd_list[[i]] <- word_vec
    U1[i, ] <- u1
    U2[i, ] <- u2
    Z1_list[[i]] <- z1
    Z21_list[[i]] <- z21
    Z22_list[[i]] <- z22
  }
  if(min(colSums(WX)) > 0) break
}

#���X�g��ϊ�
wd <- unlist(wd_list)
Z1 <- do.call(rbind, Z1_list)
Z21 <- do.call(rbind, Z21_list)
Z22 <- do.call(rbind, Z22_list)
storage.mode(Z1) <- "integer"
storage.mode(Z21) <- "integer"
storage.mode(Z22) <- "integer"
storage.mode(WX) <- "integer"

#�X�p�[�X�s��ɕϊ�
word_data <- sparseMatrix(1:f, wd, x=rep(1, f), dims=c(f, v))
word_dt <- t(word_data)

#�I�u�W�F�N�g������
rm(Z1_list); rm(Z21_list); rm(Z22_list)
rm(WX); rm(wd_list)
gc(); gc()


####�}���R�t�A�������e�J�����@��Switching Binomial LDA�𐄒�####
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
R <- 2000
keep <- 2  
iter <- 0
burnin <- 300/keep
disp <- 10

##���O���z�̐ݒ�
#�g�s�b�N���z�ƃX�C�b�`���O�ϐ��̎��O���z
alpha11 <- 0.25; alpha12 <- 0.25
alpha21 <- 0.1; alpha22 <- 0.1
beta <- 0.5

#�P�ꕪ�z�̎��O���z
alpha31 <- alpha32 <- alpha33 <- 0.1


##�p�����[�^�̐^�l
#�g�s�b�N���z�ƃX�C�b�`���O�ϐ��̐^�l
theta11 <- thetat11
theta12 <- thetat12
theta21 <- thetat21
theta22 <- thetat22
lambda <- lambdat

#�X�R�A���z�ƒP�ꕪ�z�̐^�l
eta <- etat
phi <- phit
gamma <- gammat
omega <- omegat


##�p�����[�^�̏����l��ݒ�
#�g�s�b�N���z�ƃX�C�b�`���O�ϐ��̏����l
theta11 <- extraDistr::rdirichlet(hh, rep(1.0, k11))
theta12 <- extraDistr::rdirichlet(item, rep(1.0, k12))
theta21 <- extraDistr::rdirichlet(hh, rep(1.0, k21))
theta22 <- extraDistr::rdirichlet(item, rep(1.0, k22))
lambda <- matrix(1/s, nrow=hh, ncol=s)

#�X�R�A���z�ƒP�ꕪ�z�̏����l
eta <- extraDistr::rdirichlet(k11*k12, rep(1.0, r))   #�]���X�R�A���z�̏����l
phi <- extraDistr::rdirichlet(k21, rep(2.0, v))   #���[�U�[�̒P�ꕪ�z�̏����l
gamma <- extraDistr::rdirichlet(k22, rep(2.0, v))   #�A�C�e���̒P�ꕪ�z�̏����l
omega <- extraDistr::rdirichlet(r, rep(2.0, v))   #�]���X�R�A�̒P�ꕪ�z�̏����l


##�p�����[�^�̊i�[�p�z��
#�g�s�b�N���z�ƃX�C�b�`���O�ϐ��̊i�[�p�z��
THETA11 <- array(0, dim=c(hh, k11, R/keep))
THETA12 <- array(0, dim=c(item, k12, R/keep))
THETA21 <- array(0, dim=c(hh, k21, R/keep))
THETA22 <- array(0, dim=c(item, k22, R/keep))
LAMBDA <- array(0, dim=c(hh, s, R/keep))

#�X�R�A���z�ƒP�ꕪ�z�̊i�[�p�z��
ETA <- array(0, dim=c(k11*k12, r, R/keep))
PHI <- array(0, dim=c(k21, v, R/keep))
GAMMA <- array(0, dim=c(k22, v, R/keep))
OMEGA <- array(0, dim=c(r, v, R/keep))

#�g�s�b�N�̊i�[�p�z��
U_SEG1 <- matrix(0, nrow=d, ncol=k11)
U_SEG2 <- matrix(0, nrow=d, ncol=k12)
SEG1 <- matrix(0, nrow=f, ncol=a)
SEG21 <- matrix(0, nrow=f, ncol=k21)
SEG22 <- matrix(0, nrow=f, ncol=k22)
storage.mode(U_SEG1) <- "integer"
storage.mode(U_SEG2) <- "integer"
storage.mode(SEG21) <- "integer"
storage.mode(SEG22) <- "integer"

##�f�[�^�ƃC���f�b�N�X�̐ݒ�
#���[�U�[�̃C���f�b�N�X
user_dt <- sparseMatrix(u_id, 1:f, x=rep(1, f), dims=c(hh, f))
user_n <- rowSums(user_dt)

#�A�C�e���̃C���f�b�N�X
item_dt <- sparseMatrix(i_id, 1:f, x=rep(1, f), dims=c(item, f))
item_n <- rowSums(item_dt)

#�P��̃C���f�b�N�X
wd_dt <- t(word_data)
index_k11 <- rep(1:k12, k11)
index_k12 <- rep(1:k11, rep(k12, k11))


#�f�[�^�̐ݒ�
y_vec <- y[d_id]
y_data <- matrix(as.numeric(table(1:f, y_vec)), nrow=f, ncol=r)
storage.mode(y_data) <- "integer"
r_vec <- rep(1, r)
a_vec <- rep(1, a)
vec11 <- rep(1, k11)
vec12 <- rep(1, k12)
vec21 <- rep(1, k21)
vec22 <- rep(1, k22)
K11 <- matrix(1:(k11*k12), nrow=k11, ncol=k12, byrow=T)
K12 <- matrix(1:(k11*k12), nrow=k12, ncol=k11, byrow=T)


##�ΐ��ޓx�̊�l
#���j�O�������f���̑ΐ��ޓx
LLst <- sum(word_data %*% log(colSums(word_data) / f))

#�^�l�ł̑ΐ��ޓx
Li_score <- as.numeric((t(omegat)[wd, ] * y_data) %*% r_vec)   #�X�R�A�ޓx
Li_user <- thetat21[u_id, ] * t(phit)[wd, ]; par_user <- as.numeric(Li_user %*% vec21)   #���[�U�[�ޓx
Li_item <- thetat22[i_id, ] * t(gammat)[wd, ]; par_item <- as.numeric(Li_item %*% vec22)   #�A�C�e���ޓx
par <- rowSums(Z1 * cbind(Li_score, par_user, par_item))   #���Җޓx
LLbest <- sum(log(par[which(par!=0)]))   #�ΐ��ޓx�̘a
gc(); gc()


####�M�u�X�T���v�����O�Ńp�����[�^���T���v�����O####
for(rp in 1:R){
  
  ##�]�_�X�R�A�̃��[�U�[�g�s�b�N���T���v�����O
  #���[�U�[�g�s�b�N�̏����t���m��
  eta11 <- t(eta)[y, ] * theta12[item_id, index_k11]
  par_u0 <- matrix(0, nrow=d, ncol=k11)
  for(j in 1:k11){
    par_u0[, j] <- eta11[, K1[j, ]] %*% vec11
  }
  par_u1 <- theta11[user_id, ] * par_u0   #���[�U�[�g�s�b�N�̊��Җޓx
  
  #���ݕϐ��̊����m������g�s�b�N���T���v�����O
  u1_rate <- par_u1 / as.numeric(par_u1 %*% vec11)
  Ui1 <- rmnom(d, 1, u1_rate)
  
  
  ##�]�_�X�R�A�̃A�C�e���g�s�b�N���T���v�����O
  #�A�C�e���g�s�b�N�̏����t���m��
  eta12 <- t(eta)[y, ] * theta11[item_id, index_k12]
  par_u0 <- matrix(0, nrow=d, ncol=k12)
  for(j in 1:k12){
    par_u0[, j] <- eta12[, K1[, j]] %*% vec12
  }
  par_u2 <- theta12[item_id, ] * par_u0   #�A�C�e���g�s�b�N�̊��Җޓx
  
  #���ݕϐ��̊����m������g�s�b�N���T���v�����O
  u2_rate <- par_u2 / as.numeric(par_u2 %*% vec12)
  Ui2 <- rmnom(d, 1, u2_rate)
  
  
  ##�������z���X�C�b�`���O�ϐ����T���v�����O
  #�]���X�R�A�A���[�U�[����уA�C�e���̊��Җޓx��ݒ�
  Li_score <- as.numeric((t(omega)[wd, ] * y_data) %*% r_vec)   #�X�R�A�ޓx
  Li_user <- theta21[u_id, ] * t(phi)[wd, ]; par_user <- as.numeric(Li_user %*% vec21)   #���[�U�[�ޓx
  Li_item <- theta22[i_id, ] * t(gamma)[wd, ]; par_item <- as.numeric(Li_item %*% vec22)   #�A�C�e���ޓx
  par <- cbind(Li_score, par_user, par_item)   #���Җޓx
  
  #���݊m������X�C�b�`���O�ϐ��𐶐�
  lambda_r <- lambda[u_id, ]   #�X�C�b�`���O�ϐ��̎��O���z
  par_r <- lambda_r * par
  s_prob <- par_r / as.numeric(par_r %*% a_vec)   #�X�C�b�`���O�ϐ��̊����m��
  Zi1 <- rmnom(f, 1, s_prob)   #�������z����X�C�b�`���O�ϐ��𐶐�
  index_z21 <- which(Zi1[, 2]==1)
  index_z22 <- which(Zi1[, 3]==1)
  
  #�f�B���N�����z���獬�������T���v�����O
  rsum <- as.matrix(user_dt %*% Zi1) + beta   #�f�B���N�����z�̃p�����[�^
  lambda <- extraDistr::rdirichlet(hh, rsum)   #�f�B���N�����z����lambda���T���v�����O
  
  
  ##���[�U�[����уA�C�e���̃g�s�b�N���T���v�����O
  #�g�s�b�N�̊����m���𐄒�
  z_rate1 <- Li_user[index_z21, ] / par_user[index_z21]   #���[�U�[�̃g�s�b�N�����m��
  z_rate2 <- Li_item[index_z22, ] / par_item[index_z22]   #�A�C�e���̃g�s�b�N�����m��
  
  #�������z����g�s�b�N�𐶐�
  Zi21 <- matrix(0, nrow=f, ncol=k21)
  Zi22 <- matrix(0, nrow=f, ncol=k22)
  Zi21[index_z21, ] <- rmnom(nrow(z_rate1), 1, z_rate1)
  Zi22[index_z22, ] <- rmnom(nrow(z_rate2), 1, z_rate2)

  
  ##�g�s�b�N���f���̃p�����[�^���T���v�����O
  #���[�U�[�̃g�s�b�N���z���T���v�����O
  wusum <- as.matrix(user_dt %*% Zi21) + alpha21   #�f�B���N�����z�̃p�����[�^
  theta21 <- extraDistr::rdirichlet(hh, wusum)   #�p�����[�^���T���v�����O
  
  #�A�C�e���̃g�s�b�N���z���T���v�����O
  wisum <- as.matrix(item_dt %*% Zi22) + alpha22   #�f�B���N�����z�̃p�����[�^
  theta22 <- extraDistr::rdirichlet(item, wisum)   #�p�����[�^���T���v�����O
  
  
  ##�]���X�R�A�A���[�U�[����уA�C�e���̒P�ꕪ�z���T���v�����O
  #�f�B���N�����z�̃p�����[�^
  Zi1_y <- y_data * Zi1[, 1]
  vssum <- t(as.matrix(wd_dt %*% Zi1_y)) + alpha31
  vusum <- t(as.matrix(wd_dt %*% Zi21)) + alpha32
  visum <- t(as.matrix(wd_dt %*% Zi22)) + alpha33
  
  #�f�B�N�����z����p�����[�^���T���v�����O
  omega <- extraDistr::rdirichlet(r, vssum)
  phi <- extraDistr::rdirichlet(k21, vusum)
  gamma <- extraDistr::rdirichlet(k22, visum)
  
  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^���i�[
  if(rp%%keep==0){
    #�T���v�����O���ʂ̊i�[
    mkeep <- rp/keep
    THETA21[, , mkeep] <- theta21
    THETA22[, , mkeep] <- theta22
    PHI[, , mkeep] <- phi
    GAMMA[, , mkeep] <- gamma
    OMEGA[, , mkeep] <- omega
    LAMBDA[, , mkeep] <- lambda
  }  
  
  #�g�s�b�N�����̓o�[���C�����Ԃ𒴂�����i�[����
  if(rp%%keep==0 & rp >= burnin){
    SEG1 <- SEG1 + Zi1
    SEG21 <- SEG21 + Zi21
    SEG22 <- SEG22 + Zi22
  }
  
  if(rp%%disp==0){
    #�ΐ��ޓx���v�Z
    LL <- sum(log(rowSums(Zi1 * par)))
    
    #�T���v�����O���ʂ��m�F
    print(rp)
    print(c(LL, LLbest, LLst))
    print(round(c(colMeans(Zi1), colMeans(Z1)), 3))
    print(round(cbind(phi[, (v1-5):(v1+4)], phit[, (v1-5):(v1+4)]), 3))
  }
}
