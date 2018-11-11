#####�����m���I���݈Ӗ����#####
library(MASS)
library(lda)
library(RMeCab)
detach("package:bayesm", unload=TRUE)
library(gtools)
library(reshape2)
library(dplyr)
library(plyr)
library(ggplot2)

set.seed(58079)

####�f�[�^�̔���####
#set.seed(423943)
#�����f�[�^�̐ݒ�
k <- 8   #�g�s�b�N��
d <- 2000   #������
v <- 300   #��b��
w <- rpois(d, 250)   #1����������̒P�ꐔ
a <- 6   #�⏕�ϐ���
x0 <- rpois(d, 1.5)
x <- ifelse(x0 < 1, 1, ifelse(x0 > 3, 3, x0))

#�p�����[�^�̐ݒ�
alpha0 <- round(runif(k, 0.1, 1.25), 3)   #�����̃f�B���N�����O���z�̃p�����[�^
alpha1 <- rep(0.25, v)   #�P��̃f�B���N�����O���z�̃p�����[�^
alpha2 <- rep(0.4, a)   #�⏕�f�[�^�̃f�B�N�������O���z�̃p�����[�^

#�f�B���N�������̔���
theta <- rdirichlet(d, alpha0)   #�����̃g�s�b�N���z���f�B���N���������甭��
phi <- rdirichlet(k, alpha1)   #�P��̃g�s�b�N���z���f�B���N���������甭��
omega <- rdirichlet(k, alpha2)   #�⏕�f�[�^�̃g�s�b�N���z���f�B�N�����������甭��

#�������z�̗�������f�[�^�𔭐�
WX <- matrix(0, nrow=d, ncol=v)
AX <- matrix(0, nrow=d, ncol=a)
Z1 <- list()
Z2 <- list()

for(i in 1:d){
  print(i)
  
  #�����̃g�s�b�N���z�𔭐�
  z1 <- t(rmultinom(w[i], 1, theta[i, ]))   #�����̃g�s�b�N���z�𔭐�
  
  #�����̃g�s�b�N���z����P��𔭐�
  zn <- z1 %*% c(1:k)   #0,1�𐔒l�ɒu��������
  zdn <- cbind(zn, z1)   #apply�֐��Ŏg����悤�ɍs��ɂ��Ă���
  wn <- t(apply(zdn, 1, function(x) rmultinom(1, 1, phi[x[1], ])))   #�����̃g�s�b�N����P��𐶐�
  wdn <- colSums(wn)   #�P�ꂲ�Ƃɍ��v����1�s�ɂ܂Ƃ߂�
  WX[i, ] <- wdn  
  
  #�����̃g�s�b�N���z����⏕�ϐ��𔭐�
  z2 <- t(rmultinom(x[i], 1, theta[i, ]))
  zx <- z2 %*% 1:k
  zax <- cbind(zx, z2)
  an <- t(apply(zax, 1, function(x) rmultinom(1, 1, omega[x[1], ])))
  adn <- colSums(an)
  AX[i, ] <- adn
  
  #�����g�s�b�N����ѕ⏕���g�s�b�N���i�[
  Z1[[i]] <- zdn[, 1]
  Z2[[i]] <- zax[, 1]
}

####EM�A���S���Y���Ńg�s�b�N���f���𐄒�####
####�g�s�b�N���f���̂��߂̃f�[�^�Ɗ֐��̏���####
##���ꂼ��̕������̒P��̏o������ѕ⏕���̏o�����x�N�g���ɕ��ׂ�
##�f�[�^����pID���쐬
ID1_list <- list()
wd_list <- list()
ID2_list <- list()
ad_list <- list()

#���l���Ƃɋ��lID����ђP��ID���쐬
for(i in 1:nrow(WX)){
  print(i)
  
  #�P���ID�x�N�g�����쐬
  ID1_list[[i]] <- rep(i, w[i])
  num1 <- (WX[i, ] > 0) * (1:v)
  num2 <- subset(num1, num1 > 0)
  W1 <- WX[i, (WX[i, ] > 0)]
  number <- rep(num2, W1)
  wd_list[[i]] <- number
  
  #�⏕����ID�x�N�g�����쐬
  ID2_list[[i]] <- rep(i, x[i])
  num1 <- (AX[i, ] > 0) * (1:a)
  num2 <- subset(num1, num1 > 0)
  A1 <- AX[i, (AX[i, ] > 0)]
  number <- rep(num2, A1)
  ad_list[[i]] <- number
}

#���X�g���x�N�g���ɕϊ�
ID1_d <- unlist(ID1_list)
ID2_d <- unlist(ID2_list)
wd <- unlist(wd_list)
ad <- unlist(ad_list)

##�C���f�b�N�X���쐬
doc1_list <- list()
word_list <- list()
doc2_list <- list()
aux_list <- list()
for(i in 1:length(unique(ID1_d))) {doc1_list[[i]] <- subset(1:length(ID1_d), ID1_d==i)}
for(i in 1:length(unique(wd))) {word_list[[i]] <- subset(1:length(wd), wd==i)}
for(i in 1:length(unique(ID2_d))) {doc2_list[[i]] <- subset(1:length(ID2_d), ID2_d==i)}
for(i in 1:length(unique(ad))) {aux_list[[i]] <- subset(1:length(ad), ad==i)}
gc(); gc()


##�P�ꂲ�Ƃɖޓx�ƕ��S�����v�Z����֐�
burden_fr <- function(theta, phi, wd, w, k){
  Bur <-  matrix(0, nrow=length(wd), ncol=k)   #���S�W���̊i�[�p
  for(kk in 1:k){
    #���S�W�����v�Z
    Bi <- rep(theta[, kk], w) * phi[kk, c(wd)]   #�ޓx
    Bur[, kk] <- Bi   
  }
  Br <- Bur / rowSums(Bur)   #���S���̌v�Z
  r <- colSums(Br) / sum(Br)   #�������̌v�Z
  bval <- list(Br=Br, Bur=Bur, r=r)
  return(bval)
}

####EM�A���S���Y���̏����l��ݒ肷��####
##�����l�������_���ɐݒ�
#phi�̏����l
freq_v <- matrix(colSums(WX), nrow=k, ncol=v, byrow=T)   #�P��̏o����
rand_v <- matrix(trunc(rnorm(k*v, 0, (colSums(WX)/2))), nrow=k, ncol=v, byrow=T)   #�����_����
phi_r <- abs(freq_v + rand_v) / rowSums(abs(freq_v + rand_v))   #�g�s�b�N���Ƃ̏o�����������_���ɏ�����

#theta�̏����l
theta_r <- rdirichlet(d, runif(k, 0.2, 4))   #�f�B���N�����z���珉���l��ݒ�

#omega�̏����l
freq_v <- matrix(colSums(AX)/sum(AX), nrow=k, ncol=a, byrow=T)
rand_v <- matrix(trunc(rnorm(k*a, 0, (colSums(AX)/2))), nrow=k, ncol=a, byrow=T)   #�����_����
omega_r <- abs(freq_v + rand_v) / rowSums(abs(freq_v + rand_v))   #�g�s�b�N���Ƃ̏o�����������_���ɏ�����

###�p�����[�^�̍X�V
##�P�ꃌ�x���̕��S���ƃp�����[�^��������
#�P�ꃌ�x���̕��S���̍X�V
word_fr <- burden_fr(theta=theta_r, phi=phi_r, wd=wd, w=w, k=k)
Bw <- word_fr$Br   #���S��
r1 <- word_fr$r   #������

#theta�̍X�V
wsum <- (data.frame(id=ID1_d, Br=Bw) %>%
           dplyr::group_by(id) %>%
           dplyr::summarize_each(funs(sum)))[, 2:(k+1)]
theta_r <- wsum / matrix(w, nrow=d, ncol=k)   #�p�����[�^���v�Z

##phi�̍X�V
vf <- (data.frame(id=wd, Br=Bw) %>%
         dplyr::group_by(id) %>%
         dplyr::summarize_each(funs(sum)))[, 2:(k+1)]
phi_r <- t(vf) / matrix(colSums(vf), nrow=k, ncol=v)


##�⏕��񃌃x���̕��S���ƃp�����[�^��������
#�⏕��񃌃x���̕��S���̍X�V
aux_fr <- burden_fr(theta=theta_r, phi=omega_r, wd=ad, w=x, k=k)
Ba <- aux_fr$Br   #���S��
r2 <- aux_fr$r   #������

##omega�̍X�V
af <- (data.frame(id=ad, Br=Ba) %>%
         dplyr::group_by(id) %>%
         dplyr::summarize_each(funs(sum)))[, 2:(k+1)]
omega_r <- t(af) / matrix(colSums(af), nrow=k, ncol=a)


#�ΐ��ޓx�̌v�Z
LLw <- sum(log(rowSums(word_fr$Bur)))   #�P�ꃌ�x���̑ΐ��ޓx
LLa <- sum(log(rowSums(aux_fr$Bur)))   #�⏕��񃌃x���̑ΐ��ޓx
LL <- LLw + LLa


####EM�A���S���Y���Ńp�����[�^���X�V####
#�X�V�X�e�[�^�X
iter <- 1
dl <- 100   #EM�X�e�b�v�ł̑ΐ��ޓx�̍��̏����l
tol <- 0.5
LL1 <- LL   #�ΐ��ޓx�̏����l
LLs <- c()

##EM�A���S���Y������������܂Ŕ���������
while(abs(dl) >= tol){   #dl��tol�ȏ�̏ꍇ�͌J��Ԃ�
  
  ##�P�ꃌ�x���̃p�����[�^���Ŗސ���
  #�P�ꃌ�x���̕��S���̍X�V
  word_fr <- burden_fr(theta=theta_r, phi=phi_r, wd=wd, w=w, k=k)
  Bw <- word_fr$Br   #���S��
  r1 <- word_fr$r   #������
  
  #theta�̍X�V
  wsum <- (data.frame(id=ID1_d, Br=Bw) %>%
             dplyr::group_by(id) %>%
             dplyr::summarize_each(funs(sum)))[, 2:(k+1)]
  theta_r <- wsum / matrix(w, nrow=d, ncol=k)   #�p�����[�^���v�Z
  
  ##phi�̍X�V
  vf <- (data.frame(id=wd, Br=Bw) %>%
           dplyr::group_by(id) %>%
           dplyr::summarize_each(funs(sum)))[, 2:(k+1)]
  phi_r <- t(vf) / matrix(colSums(vf), nrow=k, ncol=v)
  
  
  ##�⏕��񃌃x���̃p�����[�^���Ŗސ���
  #�⏕��񃌃x���̕��S���̍X�V
  aux_fr <- burden_fr(theta=theta_r, phi=omega_r, wd=ad, w=x, k=k)
  Ba <- aux_fr$Br   #���S��
  r2 <- aux_fr$r   #������
  
  ##omega�̍X�V
  af <- (data.frame(id=ad, Br=Ba) %>%
           dplyr::group_by(id) %>%
           dplyr::summarize_each(funs(sum)))[, 2:(k+1)]
  omega_r <- t(af) / matrix(colSums(af), nrow=k, ncol=a)
  
  
  ##�ϑ��f�[�^�̑ΐ��ޓx�̌v�Z
  LLw <- sum(log(rowSums(word_fr$Bur)))   #�P�ꃌ�x���̑ΐ��ޓx
  LLa <- sum(log(rowSums(aux_fr$Bur)))   #�⏕��񃌃x���̑ΐ��ޓx
  LL <- LLw + LLa
  
  ##�A���S���Y���̍X�V
  iter <- iter+1
  dl <- LL1 - LL
  LL1 <- LL
  LLs <- c(LLs, LL)
  print(LL)
}

####���茋�ʂƓ��v��####
plot(1:length(LLs), LLs, type="l", xlab="iter", ylab="LL", main="�ΐ��ޓx�̕ω�", lwd=2)

(PHI <- data.frame(round(t(phi_r), 3), t=round(t(phi), 3)))   #phi�̐^�̒l�Ɛ��茋�ʂ̔�r
(OMEGA <- data.frame(round(t(omega_r), 3), t=round(t(omega), 3)))   #omega�̐^�̒l�Ɛ��茋�ʂ̔�r
(THETA <- data.frame(w, round(theta_r, 3), t=round(theta, 3)))   #theta�̐^�̒l�Ɛ��茋�ʂ̔�r
r   #�������̐��茋��

round(colSums(THETA[, 2:(k+1)]) / sum(THETA[, 2:(k+1)]), 3)   #���肳�ꂽ�������̊e�g�s�b�N�̔䗦
round(colSums(THETA[, (k+1):(2*k)]) / sum(THETA[, (k+1):(2*k)]), 3)   #�^�̕������̊e�g�s�b�N�̔䗦

#AIC��BIC
tp <- dim(theta_r)[1]*dim(theta_r)[2] 
pp <- dim(phi_r)[1]*dim(phi_r)[2] + dim(omega_r)[1]*dim(omega_r)[2]

(AIC <- -2*LL + 2*(tp+pp)) 
(BIC <- -2*LL + log(nrow(WX))*(tp+pp))

##���ʂ��O���t��
#theta�̃v���b�g(50�Ԗڂ̕����܂�)
barplot(theta_r[1:100, 1], ylim=c(0, 1), col=c(1:ncol(phi_r)), density=50)
abline(h=mean(theta_r[, 1]), col=10, lty=5)
barplot(theta_r[1:100, 2], ylim=c(0, 1), col=c(1:ncol(phi_r)), density=50)
abline(h=mean(theta_r[, 2]), col=10, lty=5)
barplot(theta_r[1:100, 3], ylim=c(0, 1), col=c(1:ncol(phi_r)), density=50)
abline(h=mean(theta_r[, 2]), col=10, lty=5)
barplot(theta_r[1:100, 4], ylim=c(0, 1), col=c(1:ncol(phi_r)), density=50)
abline(h=mean(theta_r[, 2]), col=10, lty=5)
barplot(theta_r[1:100, 5], ylim=c(0, 1), col=c(1:ncol(phi_r)), density=50)
abline(h=mean(theta_r[, 2]), col=10, lty=5)

#phi�̃v���b�g(50�Ԗڂ̒P��܂�)
barplot(phi_r[1, 1:50], ylim=c(0, 0.05), col=c(1:ncol(phi_r)), density=50)
abline(h=mean(phi_r[1, ]), col=10, lty=5)
barplot(phi_r[2, 1:50], ylim=c(0, 0.05), col=c(1:ncol(phi_r)), density=50)
abline(h=mean(phi_r[2, ]), col=10, lty=5)
barplot(phi_r[3, 1:50], ylim=c(0, 0.05), col=c(1:ncol(phi_r)), density=50)
abline(h=mean(phi_r[3, ]), col=10, lty=5)
barplot(phi_r[4, 1:50], ylim=c(0, 0.05), col=c(1:ncol(phi_r)), density=50)
abline(h=mean(phi_r[4, ]), col=10, lty=5)
barplot(phi_r[5, 1:50], ylim=c(0, 0.05), col=c(1:ncol(phi_r)), density=50)
abline(h=mean(phi_r[5, ]), col=10, lty=5)

#omega�̃v���b�g
barplot(omega_r[1, ], col=c(1:ncol(omega_r)), density=50)
abline(h=mean(omega_r[1, ]), col=10, lty=5)
barplot(omega_r[2, ], col=c(1:ncol(omega_r)), density=50)
abline(h=mean(omega_r[2, ]), col=10, lty=5)
barplot(omega_r[3, ], col=c(1:ncol(omega_r)), density=50)
abline(h=mean(omega_r[3, ]), col=10, lty=5)
barplot(omega_r[4, ], col=c(1:ncol(omega_r)), density=50)
abline(h=mean(omega_r[4, ]), col=10, lty=5)
barplot(omega_r[5, ], col=c(1:ncol(omega_r)), density=50)
abline(h=mean(omega_r[5, ]), col=10, lty=5)
