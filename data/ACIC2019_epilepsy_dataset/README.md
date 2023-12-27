# A synthetic data from the 2019 American Causal Inference Conferen(ACIC2019)


We downloaded from the official website of ACIC and packaged these codes（Data Generator Processor, DGP） into a zip file: [ACIC_2019_Generate_DGPs.zip](ACIC_2019_Generate_DGPs.zip)


We generated the dataset using the method4 in the `SG_Generate_High_Dim_Binary/generate_simEpilepsy.R`   which can be found in `ACIC_2019_Generate_DGPs.zip`:
``` R
#------------------------------------------------------------------
# mod 4: do treatment heterogeneity along with  IVs.
#------------------------------------------------------------------

set.seed(40)
beta.A.mod4 <- runif(ncol(temp.mod2), -.1, .12) / apply(temp.mod2, 2, sd)
logitA.mod4 <- -.1 + temp.mod2 %*% beta.A.mod4 
beta.Y.mod4 <- 2 * beta.A.mod4
beta.Y.mod4[1:5] <- 0  # create IVs
logit.drs.mod4 <- -1.8 + as.matrix(temp.mod2) %*% beta.Y.mod4+ as.matrix(d[,c(150, 160)]) %*% c(-.005, -0.02)
set.seed(21)
# true ATE in the population
psi0.mod4 <- mean(plogis( 2 + .01 * d[,160] + logit.drs.mod4) - plogis(logit.drs.mod4))    # 0.2916274
# Generate datasets
d.mod4 <- data.frame(Y= NA, A= NA, d)
set.seed(4)
niter <- 100
n.b <- 2000
for (i in mod4_files){
	b <- sample(1:n, n.b, replace = TRUE)
	d.mod4$A[b] <-rbinom(n.b, 1, plogis(logitA.mod4)[b])
	d.mod4$Y[b]<-  rbinom(n.b, 1, plogis(d.mod4$A * (2 + .01 * d[,160]) + logit.drs.mod4)[b])
	write.csv(d.mod4[b,], file =  paste0("epilepsyMod4", i , ".csv"), row.names = FALSE)
}

```

