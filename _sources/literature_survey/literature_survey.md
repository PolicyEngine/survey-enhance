# Literature survey

## Introduction

Over the last few decades, there has been extensive research into the
accuracy of household surveys for estimating socioeconomic and
policy-related indicators, as well as methods of improving survey
accuracy. Most of these studies have focussed on one particular
mechanism by which surveys introduce inaccuracy (for example, by
omitting top incomes or under-sampling low incomes), and examine a
method of improving surveys which tackles this particular flaw. This
literature survey aims to provide a comprehensive overview of the state
of the art in improving survey accuracy, while also examining how and if
these individual advancements complement each other.

## Current approaches in economic surveys

It is well known that household surveys produce inconsistent results to
other data sources, such as administrative databases. Given the nature
of how surveys are conducted (households must first consent to an
interview, and secondarily must answer truthfully to questions asked),
this inaccuracy can be introduced either by sampling error or
measurement error (likely both, to some extent). Over the last few
decades, household surveys have become the dominant tool in measuring
and projecting economic impacts of policy changes, and as such, there
has been a great deal of research into improving survey accuracy.

### Under-coverage of high incomes

The Department for Work and Pensions is required by law to report on
poverty and inequality metrics every year, and in meeting this
requirement, it publishes a household-level dataset of disposable
incomes, termed the Households Below Average Income (HBAI) dataset.{cite:ps}`hbai` 
Since 1992, it has applied an adjustment to the disposable incomes of a
subset of the dataset in order to make the coverage of top incomes more
comparable with that of HMRC’s Survey of Personal Incomes (SPI) dataset
-- this adjustment termed the ‘SPI adjustment’. In {cite:ps}`ifs_survey_under_coverage`, the authors examine
the methodology of this adjustment, as well as its performance against
its original goals.

The authors document[^1] the steps of the SPI adjustment, which involve
first identifying a set of ‘rich’ households. The definition of rich
applies a condition that a household’s income must be above a certain
threshold, where separate thresholds are used for pensioner and
non-pensioner households. The target used to set thresholds is generally
to ensure that around 0.5% of records are altered, varying by year. The
HBAI ‘rich’ households are then modified by replacing gross incomes (an
income measure which the SPI also contains) with the average values for
records in the same group in the SPI. Finally, the survey weights are
recalculated: in the original survey, weights are solved by matching
administrative statistics on population sizes; under the SPI adjustment,
population sizes of the ‘rich’ groups are included in the set of
statistical targets to hit. The authors find that the SPI adjustment has
been successful in improving the coverage of top incomes in the HBAI
dataset, but raise a number of issues:

#### Income decomposition

The SPI adjustment is applied to a singular income variable, but the FRS
contains a number of components. Modifying gross income, but not
modifying employment income, savings income, etc. breaks the link
between these variables, which prevents researchers from conducting
decomposition analyses.

#### Stratification

There is no obvious justification for separate thresholds for pensioners
and non-pensioners (and further, between households in Great Britain and
Northern Ireland). The authors suggest these stratification choices were
made in order to minimise methodological changes over time, for example
as the survey expanded to Northern Ireland.

#### SPI lag

The Survey of Personal Incomes is not routinely available at the same
time as the Family Resources Survey (from which the HBAI microdata is
derived). Therefore the SPI adjustment is applied to the HBAI dataset
using a lagged SPI dataset, which may introduce additional inaccuracy.

### Adjustments using administrative tax data

For the 2019 edition of the Households Below Average Income series, the
ONS published details of the methodology used to tune the dataset with the SPI in {cite:ps}`ons_spi_version_2`. They respond to some of the concerns raised by {cite:ps}`ifs_survey_under_coverage`:

#### Pensioner stratification

The authors show that high-income pensioners and non-pensioners are both
under-represented in their respective populations but comparing the
ratios of incomes at different quantiles, finding that a common
threshold for both groups would fail to ensure that pensioners (who have
lower income, on average) are sufficiently affected by the SPI
adjustment.

#### Choice of income threshold

The authors discuss possible justifications for a particular income
threshold, mostly based on the quantile at which divergence between the
FRS and SPI ‘became an issue’. However, the choice to use a binary
variable (rather than, for example, phasing in an SPI adjustment) here
is arbitrary, and the authors do not address the reasons why this choice
was made.

#### SPI lag

The authors acknowledge the issue of using SPI projections, rather than
actual outturn data, and examine the size of this effect. They find that
revising recent survey releases with the actual SPI data later released
changed the Gini coefficient of income inequality estimates by around
0.2 percentage points. This is considered to be small and therefore
recommend against the need for the ONS to re-publish statistics when
current SPI data becomes available.

### Capital income imputation

The issue of income decomposition remained largely untackled until {cite:ps}`frs_capital_income`, in
which the authors attempt to improve the reporting of a specific
component of gross income which is more severely under-reported in the
FRS than others: capital income. They first establish that income
under-reporting is mostly due to this particular category by comparing
individual income sources between the FRS and SPI, finding that the
aggregates of non-capital income are around 100% of the totals for the
SPI, while capital income is only around 40% as represented.

The authors present a novel observation about the instances where
capital income is under-reported: the capital share of income in
individuals is far less represented in the FRS than in the SPI
(specifically, the number of individuals with a ‘high capital share’),
rather than simply a lack of high-capital-income individuals. They
introduce a new method to correct for this under-capture: adjust the
weights of high-capital-share individuals in order to match the totals
in SPI data.

The authors find that the new method is largely successful at correcting
for under-capture of capital income, and increases the Gini coefficient
of FRS data by between 2 and 5 percentage points (applying the
methodology to historical FRS data releases). However, they do not
measure the changes to how well the FRS ranks against other aspects of
the SPI.

### Under-coverage of very low incomes

In {cite:ps}`brewer_low_income_coverage`, the authors examine the other end of the income spectrum, finding
that very low-income households tend to spend much more than moderately
low-income households in the Living Cost and Food Survey (a household
survey with similar administration to the FRS). The authors report a
variety of evidence that income at the low end is misreported in the
survey:

#### Missing benefit spending

By comparing total reported receipt of benefits by recipients with
aggregate spending figures published by the DWP and HMRC, the authors
find that the FRS and LCFS consistently under-report benefit income by
around 5%, and that this figure has become worse over the last decade,
rising from 2.5% in 2000.

#### Sub-minimum wage reporting

In the LCFS, individuals report both hours worked and annual earnings,
enabling researchers to calculate the implied hourly wage. For 10.5% of
individuals in 2009, this was below the legal minimum wage. Although
this does not guarantee a breach of employment law,[^2] the proportion
is substantial and implies that either earnings are under-reported or
hours worked are over-reported.

#### High spending ratios

The authors use a model of consumption smoothing to determine whether
the overly high spending (compared to income) for low-income households
can be explained by lifetime consumption smoothing, but find that this
is not the case.

### Linking data directly to administrative data

All of the previously covered research into survey inaccuracy has
identified a common question: how much of the survey error is due to
non-response bias, and how much is due to measurement error? In {cite:ps}`dwp_110`, the
authors attempt to quantify the measurement error of the FRS by linking
individual households with data from the DWP’s administrative records,
using non-public identifiers. The process of linking is not perfect:
respondents are asked for permission to link their survey data with
administrative data, and some (around 30%) refuse. However, for each
benefit, the authors were able to find the percentage of reporting
adults for whom a link to an administrative data record could be
identified, the percentage of reporting adults recipients for whom no
link could be found, and the percentage of adults represented only by
administrative data.

The authors find that these splits vary significantly by benefit:
recipient data on the State Pension (SP) is highly accurate in the FRS
(96% of SP reported recipients were represented by the FRS, 1% were only
on the FRS and not on administrative datasets, and 3% were only on
administrative datasets). At the same time, around 62% of adults on the
FRS who reported receiving Severe Disablement Allowance could not be
identified in administrative data. There are multiple possible reasons
for this, and they vary by benefit: the recipient population is often
confused or mistaken when answering questions about their benefits, and
this is more acute for age- or disability-related benefits. This appears
to provide additional evidence that measurement error is significant, at
least at the low-income subset of the surveys.

### Linear programming

Linear programming, a mathematical technique for solving linearly
constrained optimisation problems, is commonly used to determine survey
weight values, where the criteria are defined maximum deviations from
top-level demographic statistics. In {cite:ps}`frs_weighting_review`, linear programming methods are
used to determine the optimal weights for the Family Resources Survey,
according to limits on how far apart the FRS aggregates can be from
national and regional population estimates. In both of {cite:ps}`tpc_microsim_docs` and {cite:ps}`taxdata_github`, tax models
apply a linear programming algorithm to solve for weight adjustments
satisfying a combination of tax statistic deviation constraints, and
weight adjustment magnitude limits.

## Applicable machine learning techniques

There are several reasons why machine learning techniques are
well-suited to the task of survey imputation. The most fundamental
justification is in its context-agnostic nature: machine learning
approaches do not require assumptions specific to the field they are
applied in, unlike the current approaches to survey accuracy improvement
(for example, the percentile adjustment methodology in {cite:ps}`ons_spi_version_2`, which
explicitly partitions households into ‘rich’ and ‘non-rich’ using
arguably arbitrary definitions). In other domains, for example image
classificaion, a move away from prescriptive methods towards loss
function minimisation has seen substantially improved accuracy and
robustness.{cite:ps}`image_classification_survey`

### Gradient descent

Gradient descent is a technique for finding parameters which minimise a
loss function, by iteratively updating the parameters in the direction
of the steepest negative gradient.{cite:ps}`gradient_descent` This is a highly common technique in
machine learning, and is used in a variety of contexts, most notably as
the foundation for training artificial neural networks. It relies on no
domain-specific assumptions other than those present in the definition
of the loss function, enabling it to be applied to a wide range of
problems.

Several variations of gradient descent have emerged over the years which
achieve more efficient training procedures: stochastic gradient descent
steps in the direction of an *estimate* of the gradient using individual
training examples, rather than loading the full dataset.{cite:ps}`sgd` Mini-batch
gradient descent represents a compromise between batch (full-dataset)
and stochastic gradient descent, by iterating parameters using
fixed-size subsets of the training data.{cite:ps}`mini_batch`

As well as gradient calculation methods, optimisation algorithms have
revealed significant accuracy and efficiency improvements by defining
behaviours for hyper-parameters such as the learning rate (the velocity
at which parameters follow the gradient). These include Adam,{cite:ps}`adam` AdaGrad,{cite:ps}`adagrad`
and RMSProp.

Gradient descent could feasibly be applied to survey accuracy problems,
since it requires only a loss function that is differentiable with
respect to the parameters being optimised. In the context of survey
accuracy, a loss function could be defined as the squared errors of
individual aggregate statistics between official sources, and a survey,
which would be continuously differentiable over the weights of
individual household records.

### Random forest models

Random forest models are a type of ensemble learning technique, which
combine the predictions of multiple decision trees to produce a more
accurate prediction than any individual tree.{cite:ps}`random_forests` The decision trees are
trained on a subset of the training data, and the predictions of each
tree are combined using a voting system. Although its introduction is
far less recent than more modern innovations in the field of neural
networks (for example, artificial neural network variants{cite:ps}`anns` or
transformers{cite:ps}`transformers`), random forest models have shown consistently high
accuracy across a wide range of domains, remaining competitive with the
most recent techniques.

This type of model has been applied (to a limited extent) in the context
of policy analysis, and have shown superior performance in prediction
tasks to logit and other model types.{cite:ps}`ecb`

There are several reasons why random forest models might outperform
neural networks in predicting survey microdata values from other
attributes (for example, predicting employment income from demographic
variables), but the most natural reason is that tax-benefit law, which
heavily influences financial decisions, is more similar in structure to
a random forest than a neural network. For example, in {cite:ps}`cg_bunching` the authors
found that capital gains variables are ‘unnaturally’ distributed in
order to respond to incentives set by particular tax law parameters.

## Conclusion

Current methods of enhancing surveys are largely effective at improving
the accuracy of survey data on narrowly-defined subdomains (such as
high-income analysis), but rely on explicit assumptions and are often
not completely successful at bringing household surveys to the same
level of accuracy as administrative data, especially at the low end of
the income spectrum. Machine learning techniques such as random forest
model imputation and gradient descent have shown promise in adjacent
fields to public policy analysis, and could serve as more generalisable
and effective replacements for the existing survey improvement methods.

The body of research on current methods for improving survey data is
useful for examining how effective specific approaches were in improving
a survey’s answer to a narrow domain (for example, how adjusting income
values improved the Gini index of income inequality), but there is
little research on how each of the current methods, and any of the
machine learning methods presented here, affects the overall picture of
accuracy for a survey data. The reason for this is that many accuracy
goals are orthogonal to each other: for example, improving the coverage
of high taxable incomes might improve a survey’s estimate of total
income tax liabilities, but if it achieves this by overestimating
employment income compared to dividend income, then a survey’s estimates
of payroll and dividend tax liabilities might each separately be
reduced.

An implementation a general survey accuracy loss function that takes
into account all (or as many as is feasibly possible in the scope of
this project) of these accuracy targets, as well as implementations of
both current and potential methods of data manipulation, would allow for
a more comprehensive comparison of the effectiveness of each method.

[^1]: Previously, the DWP had not published its research underlying the methodology of the SPI adjustment.

[^2]: Employers can count some in-kind benefits as payment towards the minimum wage, and there are other legal exceptions.

```{bibliography}
:style: unsrt
```
