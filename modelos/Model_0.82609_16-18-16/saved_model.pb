йГ	
ѓ
Щ

B
AssignVariableOp
resource
value"dtype"
dtypetype
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18аж

conv2d_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_45/kernel
}
$conv2d_45/kernel/Read/ReadVariableOpReadVariableOpconv2d_45/kernel*&
_output_shapes
:*
dtype0
t
conv2d_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_45/bias
m
"conv2d_45/bias/Read/ReadVariableOpReadVariableOpconv2d_45/bias*
_output_shapes
:*
dtype0

conv2d_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_46/kernel
}
$conv2d_46/kernel/Read/ReadVariableOpReadVariableOpconv2d_46/kernel*&
_output_shapes
:*
dtype0
t
conv2d_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_46/bias
m
"conv2d_46/bias/Read/ReadVariableOpReadVariableOpconv2d_46/bias*
_output_shapes
:*
dtype0

conv2d_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_47/kernel
}
$conv2d_47/kernel/Read/ReadVariableOpReadVariableOpconv2d_47/kernel*&
_output_shapes
: *
dtype0
t
conv2d_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_47/bias
m
"conv2d_47/bias/Read/ReadVariableOpReadVariableOpconv2d_47/bias*
_output_shapes
: *
dtype0
{
dense_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 * 
shared_namedense_59/kernel
t
#dense_59/kernel/Read/ReadVariableOpReadVariableOpdense_59/kernel*
_output_shapes
:	 *
dtype0
r
dense_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_59/bias
k
!dense_59/bias/Read/ReadVariableOpReadVariableOpdense_59/bias*
_output_shapes
: *
dtype0
z
dense_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @* 
shared_namedense_60/kernel
s
#dense_60/kernel/Read/ReadVariableOpReadVariableOpdense_60/kernel*
_output_shapes

: @*
dtype0
r
dense_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_60/bias
k
!dense_60/bias/Read/ReadVariableOpReadVariableOpdense_60/bias*
_output_shapes
:@*
dtype0
z
dense_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_61/kernel
s
#dense_61/kernel/Read/ReadVariableOpReadVariableOpdense_61/kernel*
_output_shapes

:@*
dtype0
r
dense_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_61/bias
k
!dense_61/bias/Read/ReadVariableOpReadVariableOpdense_61/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d_45/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_45/kernel/m

+Adam/conv2d_45/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_45/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_45/bias/m
{
)Adam/conv2d_45/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_46/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_46/kernel/m

+Adam/conv2d_46/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_46/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_46/bias/m
{
)Adam/conv2d_46/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_47/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_47/kernel/m

+Adam/conv2d_47/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2d_47/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_47/bias/m
{
)Adam/conv2d_47/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/bias/m*
_output_shapes
: *
dtype0

Adam/dense_59/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *'
shared_nameAdam/dense_59/kernel/m

*Adam/dense_59/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/m*
_output_shapes
:	 *
dtype0

Adam/dense_59/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_59/bias/m
y
(Adam/dense_59/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/m*
_output_shapes
: *
dtype0

Adam/dense_60/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*'
shared_nameAdam/dense_60/kernel/m

*Adam/dense_60/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_60/kernel/m*
_output_shapes

: @*
dtype0

Adam/dense_60/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_60/bias/m
y
(Adam/dense_60/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_60/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_61/kernel/m

*Adam/dense_61/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/m*
_output_shapes

:@*
dtype0

Adam/dense_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_61/bias/m
y
(Adam/dense_61/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_45/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_45/kernel/v

+Adam/conv2d_45/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_45/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_45/bias/v
{
)Adam/conv2d_45/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_46/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_46/kernel/v

+Adam/conv2d_46/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_46/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_46/bias/v
{
)Adam/conv2d_46/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_47/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_47/kernel/v

+Adam/conv2d_47/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_47/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_47/bias/v
{
)Adam/conv2d_47/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/bias/v*
_output_shapes
: *
dtype0

Adam/dense_59/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *'
shared_nameAdam/dense_59/kernel/v

*Adam/dense_59/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/v*
_output_shapes
:	 *
dtype0

Adam/dense_59/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_59/bias/v
y
(Adam/dense_59/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/v*
_output_shapes
: *
dtype0

Adam/dense_60/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*'
shared_nameAdam/dense_60/kernel/v

*Adam/dense_60/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_60/kernel/v*
_output_shapes

: @*
dtype0

Adam/dense_60/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_60/bias/v
y
(Adam/dense_60/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_60/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_61/kernel/v

*Adam/dense_61/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/v*
_output_shapes

:@*
dtype0

Adam/dense_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_61/bias/v
y
(Adam/dense_61/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
жK
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*K
valueKBK B§J

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
 trainable_variables
!	keras_api
R
"	variables
#regularization_losses
$trainable_variables
%	keras_api
h

&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
R
,	variables
-regularization_losses
.trainable_variables
/	keras_api
R
0	variables
1regularization_losses
2trainable_variables
3	keras_api
R
4	variables
5regularization_losses
6trainable_variables
7	keras_api
h

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
h

>kernel
?bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
h

Dkernel
Ebias
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
А
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratemmmm&m'm8m9m>m?mDm EmЁvЂvЃvЄvЅ&vІ'vЇ8vЈ9vЉ>vЊ?vЋDvЌEv­
V
0
1
2
3
&4
'5
86
97
>8
?9
D10
E11
 
V
0
1
2
3
&4
'5
86
97
>8
?9
D10
E11
­

Olayers
Pmetrics
	variables
Qlayer_metrics
regularization_losses
Rlayer_regularization_losses
Snon_trainable_variables
trainable_variables
 
\Z
VARIABLE_VALUEconv2d_45/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_45/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

Tlayers
Umetrics
	variables
Vlayer_metrics
regularization_losses
Wlayer_regularization_losses
Xnon_trainable_variables
trainable_variables
\Z
VARIABLE_VALUEconv2d_46/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_46/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

Ylayers
Zmetrics
	variables
[layer_metrics
regularization_losses
\layer_regularization_losses
]non_trainable_variables
trainable_variables
 
 
 
­

^layers
_metrics
	variables
`layer_metrics
regularization_losses
alayer_regularization_losses
bnon_trainable_variables
 trainable_variables
 
 
 
­

clayers
dmetrics
"	variables
elayer_metrics
#regularization_losses
flayer_regularization_losses
gnon_trainable_variables
$trainable_variables
\Z
VARIABLE_VALUEconv2d_47/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_47/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1
 

&0
'1
­

hlayers
imetrics
(	variables
jlayer_metrics
)regularization_losses
klayer_regularization_losses
lnon_trainable_variables
*trainable_variables
 
 
 
­

mlayers
nmetrics
,	variables
olayer_metrics
-regularization_losses
player_regularization_losses
qnon_trainable_variables
.trainable_variables
 
 
 
­

rlayers
smetrics
0	variables
tlayer_metrics
1regularization_losses
ulayer_regularization_losses
vnon_trainable_variables
2trainable_variables
 
 
 
­

wlayers
xmetrics
4	variables
ylayer_metrics
5regularization_losses
zlayer_regularization_losses
{non_trainable_variables
6trainable_variables
[Y
VARIABLE_VALUEdense_59/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_59/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91
 

80
91
Ў

|layers
}metrics
:	variables
~layer_metrics
;regularization_losses
layer_regularization_losses
non_trainable_variables
<trainable_variables
[Y
VARIABLE_VALUEdense_60/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_60/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

>0
?1
 

>0
?1
В
layers
metrics
@	variables
layer_metrics
Aregularization_losses
 layer_regularization_losses
non_trainable_variables
Btrainable_variables
[Y
VARIABLE_VALUEdense_61/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_61/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1
 

D0
E1
В
layers
metrics
F	variables
layer_metrics
Gregularization_losses
 layer_regularization_losses
non_trainable_variables
Htrainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
N
0
1
2
3
4
5
6
7
	8

9
10

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

total

count
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
}
VARIABLE_VALUEAdam/conv2d_45/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_45/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_46/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_46/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_47/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_47/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_59/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_59/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_60/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_60/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_61/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_61/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_45/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_45/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_46/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_46/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_47/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_47/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_59/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_59/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_60/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_60/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_61/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_61/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_conv2d_45_inputPlaceholder*/
_output_shapes
:џџџџџџџџџ00*
dtype0*$
shape:џџџџџџџџџ00

StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_45_inputconv2d_45/kernelconv2d_45/biasconv2d_46/kernelconv2d_46/biasconv2d_47/kernelconv2d_47/biasdense_59/kerneldense_59/biasdense_60/kerneldense_60/biasdense_61/kerneldense_61/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_864236
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
z
StaticRegexFullMatchStaticRegexFullMatchsaver_filename"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
\
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part

Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_72c28c7b19b94219a7d580f10461534d/part
h
SelectSelectStaticRegexFullMatchConst_1Const_2"/device:CPU:**
T0*
_output_shapes
: 
`

StringJoin
StringJoinsaver_filenameSelect"/device:CPU:**
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
Ѓ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ь
valueТBП.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
Щ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 

SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices$conv2d_45/kernel/Read/ReadVariableOp"conv2d_45/bias/Read/ReadVariableOp$conv2d_46/kernel/Read/ReadVariableOp"conv2d_46/bias/Read/ReadVariableOp$conv2d_47/kernel/Read/ReadVariableOp"conv2d_47/bias/Read/ReadVariableOp#dense_59/kernel/Read/ReadVariableOp!dense_59/bias/Read/ReadVariableOp#dense_60/kernel/Read/ReadVariableOp!dense_60/bias/Read/ReadVariableOp#dense_61/kernel/Read/ReadVariableOp!dense_61/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_45/kernel/m/Read/ReadVariableOp)Adam/conv2d_45/bias/m/Read/ReadVariableOp+Adam/conv2d_46/kernel/m/Read/ReadVariableOp)Adam/conv2d_46/bias/m/Read/ReadVariableOp+Adam/conv2d_47/kernel/m/Read/ReadVariableOp)Adam/conv2d_47/bias/m/Read/ReadVariableOp*Adam/dense_59/kernel/m/Read/ReadVariableOp(Adam/dense_59/bias/m/Read/ReadVariableOp*Adam/dense_60/kernel/m/Read/ReadVariableOp(Adam/dense_60/bias/m/Read/ReadVariableOp*Adam/dense_61/kernel/m/Read/ReadVariableOp(Adam/dense_61/bias/m/Read/ReadVariableOp+Adam/conv2d_45/kernel/v/Read/ReadVariableOp)Adam/conv2d_45/bias/v/Read/ReadVariableOp+Adam/conv2d_46/kernel/v/Read/ReadVariableOp)Adam/conv2d_46/bias/v/Read/ReadVariableOp+Adam/conv2d_47/kernel/v/Read/ReadVariableOp)Adam/conv2d_47/bias/v/Read/ReadVariableOp*Adam/dense_59/kernel/v/Read/ReadVariableOp(Adam/dense_59/bias/v/Read/ReadVariableOp*Adam/dense_60/kernel/v/Read/ReadVariableOp(Adam/dense_60/bias/v/Read/ReadVariableOp*Adam/dense_61/kernel/v/Read/ReadVariableOp(Adam/dense_61/bias/v/Read/ReadVariableOpConst"/device:CPU:0*<
dtypes2
02.	

&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
o
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0
i
IdentityIdentitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
І
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ь
valueТBП.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
Ь
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ј
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*Ю
_output_shapesЛ
И::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOpAssignVariableOpconv2d_45/kernel
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_1AssignVariableOpconv2d_45/bias
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_2AssignVariableOpconv2d_46/kernel
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_3AssignVariableOpconv2d_46/bias
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_4AssignVariableOpconv2d_47/kernel
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_5AssignVariableOpconv2d_47/bias
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_6AssignVariableOpdense_59/kernel
Identity_7"/device:CPU:0*
dtype0
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_7AssignVariableOpdense_59/bias
Identity_8"/device:CPU:0*
dtype0
U

Identity_9IdentityRestoreV2:8"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_8AssignVariableOpdense_60/kernel
Identity_9"/device:CPU:0*
dtype0
V
Identity_10IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_9AssignVariableOpdense_60/biasIdentity_10"/device:CPU:0*
dtype0
W
Identity_11IdentityRestoreV2:10"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_10AssignVariableOpdense_61/kernelIdentity_11"/device:CPU:0*
dtype0
W
Identity_12IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_11AssignVariableOpdense_61/biasIdentity_12"/device:CPU:0*
dtype0
W
Identity_13IdentityRestoreV2:12"/device:CPU:0*
T0	*
_output_shapes
:
[
AssignVariableOp_12AssignVariableOp	Adam/iterIdentity_13"/device:CPU:0*
dtype0	
W
Identity_14IdentityRestoreV2:13"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_13AssignVariableOpAdam/beta_1Identity_14"/device:CPU:0*
dtype0
W
Identity_15IdentityRestoreV2:14"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_14AssignVariableOpAdam/beta_2Identity_15"/device:CPU:0*
dtype0
W
Identity_16IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_15AssignVariableOp
Adam/decayIdentity_16"/device:CPU:0*
dtype0
W
Identity_17IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
d
AssignVariableOp_16AssignVariableOpAdam/learning_rateIdentity_17"/device:CPU:0*
dtype0
W
Identity_18IdentityRestoreV2:17"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_17AssignVariableOptotalIdentity_18"/device:CPU:0*
dtype0
W
Identity_19IdentityRestoreV2:18"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_18AssignVariableOpcountIdentity_19"/device:CPU:0*
dtype0
W
Identity_20IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_19AssignVariableOptotal_1Identity_20"/device:CPU:0*
dtype0
W
Identity_21IdentityRestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_20AssignVariableOpcount_1Identity_21"/device:CPU:0*
dtype0
W
Identity_22IdentityRestoreV2:21"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_21AssignVariableOpAdam/conv2d_45/kernel/mIdentity_22"/device:CPU:0*
dtype0
W
Identity_23IdentityRestoreV2:22"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_22AssignVariableOpAdam/conv2d_45/bias/mIdentity_23"/device:CPU:0*
dtype0
W
Identity_24IdentityRestoreV2:23"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_23AssignVariableOpAdam/conv2d_46/kernel/mIdentity_24"/device:CPU:0*
dtype0
W
Identity_25IdentityRestoreV2:24"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_24AssignVariableOpAdam/conv2d_46/bias/mIdentity_25"/device:CPU:0*
dtype0
W
Identity_26IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_25AssignVariableOpAdam/conv2d_47/kernel/mIdentity_26"/device:CPU:0*
dtype0
W
Identity_27IdentityRestoreV2:26"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_26AssignVariableOpAdam/conv2d_47/bias/mIdentity_27"/device:CPU:0*
dtype0
W
Identity_28IdentityRestoreV2:27"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_27AssignVariableOpAdam/dense_59/kernel/mIdentity_28"/device:CPU:0*
dtype0
W
Identity_29IdentityRestoreV2:28"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_28AssignVariableOpAdam/dense_59/bias/mIdentity_29"/device:CPU:0*
dtype0
W
Identity_30IdentityRestoreV2:29"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_29AssignVariableOpAdam/dense_60/kernel/mIdentity_30"/device:CPU:0*
dtype0
W
Identity_31IdentityRestoreV2:30"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_30AssignVariableOpAdam/dense_60/bias/mIdentity_31"/device:CPU:0*
dtype0
W
Identity_32IdentityRestoreV2:31"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_31AssignVariableOpAdam/dense_61/kernel/mIdentity_32"/device:CPU:0*
dtype0
W
Identity_33IdentityRestoreV2:32"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_32AssignVariableOpAdam/dense_61/bias/mIdentity_33"/device:CPU:0*
dtype0
W
Identity_34IdentityRestoreV2:33"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_33AssignVariableOpAdam/conv2d_45/kernel/vIdentity_34"/device:CPU:0*
dtype0
W
Identity_35IdentityRestoreV2:34"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_34AssignVariableOpAdam/conv2d_45/bias/vIdentity_35"/device:CPU:0*
dtype0
W
Identity_36IdentityRestoreV2:35"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_35AssignVariableOpAdam/conv2d_46/kernel/vIdentity_36"/device:CPU:0*
dtype0
W
Identity_37IdentityRestoreV2:36"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_36AssignVariableOpAdam/conv2d_46/bias/vIdentity_37"/device:CPU:0*
dtype0
W
Identity_38IdentityRestoreV2:37"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_37AssignVariableOpAdam/conv2d_47/kernel/vIdentity_38"/device:CPU:0*
dtype0
W
Identity_39IdentityRestoreV2:38"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_38AssignVariableOpAdam/conv2d_47/bias/vIdentity_39"/device:CPU:0*
dtype0
W
Identity_40IdentityRestoreV2:39"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_39AssignVariableOpAdam/dense_59/kernel/vIdentity_40"/device:CPU:0*
dtype0
W
Identity_41IdentityRestoreV2:40"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_40AssignVariableOpAdam/dense_59/bias/vIdentity_41"/device:CPU:0*
dtype0
W
Identity_42IdentityRestoreV2:41"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_41AssignVariableOpAdam/dense_60/kernel/vIdentity_42"/device:CPU:0*
dtype0
W
Identity_43IdentityRestoreV2:42"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_42AssignVariableOpAdam/dense_60/bias/vIdentity_43"/device:CPU:0*
dtype0
W
Identity_44IdentityRestoreV2:43"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_43AssignVariableOpAdam/dense_61/kernel/vIdentity_44"/device:CPU:0*
dtype0
W
Identity_45IdentityRestoreV2:44"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_44AssignVariableOpAdam/dense_61/bias/vIdentity_45"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
В
Identity_46Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: 
ф6

I__inference_sequential_15_layer_call_and_return_conditional_losses_864078
conv2d_45_input,
(conv2d_45_conv2d_readvariableop_resource-
)conv2d_45_biasadd_readvariableop_resource,
(conv2d_46_conv2d_readvariableop_resource-
)conv2d_46_biasadd_readvariableop_resource,
(conv2d_47_conv2d_readvariableop_resource-
)conv2d_47_biasadd_readvariableop_resource+
'dense_59_matmul_readvariableop_resource,
(dense_59_biasadd_readvariableop_resource+
'dense_60_matmul_readvariableop_resource,
(dense_60_biasadd_readvariableop_resource+
'dense_61_matmul_readvariableop_resource,
(dense_61_biasadd_readvariableop_resource
identityГ
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_45/Conv2D/ReadVariableOpЫ
conv2d_45/Conv2DConv2Dconv2d_45_input'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..*
paddingVALID*
strides
2
conv2d_45/Conv2DЊ
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_45/BiasAdd/ReadVariableOpА
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..2
conv2d_45/BiasAdd~
conv2d_45/ReluReluconv2d_45/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ..2
conv2d_45/ReluГ
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_46/Conv2D/ReadVariableOpи
conv2d_46/Conv2DConv2Dconv2d_45/Relu:activations:0'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ,,*
paddingVALID*
strides
2
conv2d_46/Conv2DЊ
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_46/BiasAdd/ReadVariableOpА
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ,,2
conv2d_46/BiasAdd~
conv2d_46/ReluReluconv2d_46/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ,,2
conv2d_46/ReluЪ
max_pooling2d_41/MaxPoolMaxPoolconv2d_46/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_41/MaxPool
dropout_27/IdentityIdentity!max_pooling2d_41/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout_27/IdentityГ
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_47/Conv2D/ReadVariableOpи
conv2d_47/Conv2DConv2Ddropout_27/Identity:output:0'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv2d_47/Conv2DЊ
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_47/BiasAdd/ReadVariableOpА
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d_47/BiasAdd~
conv2d_47/ReluReluconv2d_47/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d_47/ReluЪ
max_pooling2d_42/MaxPoolMaxPoolconv2d_47/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ

 *
ksize
*
paddingVALID*
strides
2
max_pooling2d_42/MaxPool
dropout_28/IdentityIdentity!max_pooling2d_42/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2
dropout_28/Identityu
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
flatten_15/Const
flatten_15/ReshapeReshapedropout_28/Identity:output:0flatten_15/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten_15/ReshapeЉ
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02 
dense_59/MatMul/ReadVariableOpЃ
dense_59/MatMulMatMulflatten_15/Reshape:output:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_59/MatMulЇ
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_59/BiasAdd/ReadVariableOpЅ
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_59/BiasAdds
dense_59/ReluReludense_59/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_59/ReluЈ
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02 
dense_60/MatMul/ReadVariableOpЃ
dense_60/MatMulMatMuldense_59/Relu:activations:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_60/MatMulЇ
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_60/BiasAdd/ReadVariableOpЅ
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_60/BiasAdds
dense_60/ReluReludense_60/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_60/ReluЈ
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_61/MatMul/ReadVariableOpЃ
dense_61/MatMulMatMuldense_60/Relu:activations:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_61/MatMulЇ
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_61/BiasAdd/ReadVariableOpЅ
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_61/BiasAdd|
dense_61/SoftmaxSoftmaxdense_61/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_61/Softmaxn
IdentityIdentitydense_61/Softmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00:::::::::::::` \
/
_output_shapes
:џџџџџџџџџ00
)
_user_specified_nameconv2d_45_input
Ђ	

$__inference_signature_wrapper_864236
conv2d_45_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallconv2d_45_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_8637052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:џџџџџџџџџ00
)
_user_specified_nameconv2d_45_input
Љ
J
+__inference_dropout_28_layer_call_fn_864365

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

 *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ

 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ

 :W S
/
_output_shapes
:џџџџџџџџџ

 
 
_user_specified_nameinputs
Ќ
Ќ
D__inference_dense_59_layer_call_and_return_conditional_losses_864393

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
№

*__inference_conv2d_45_layer_call_fn_864258

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ..2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ..2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ00:::W S
/
_output_shapes
:џџџџџџџџџ00
 
_user_specified_nameinputs
ћD

!__inference__wrapped_model_863705
conv2d_45_input:
6sequential_15_conv2d_45_conv2d_readvariableop_resource;
7sequential_15_conv2d_45_biasadd_readvariableop_resource:
6sequential_15_conv2d_46_conv2d_readvariableop_resource;
7sequential_15_conv2d_46_biasadd_readvariableop_resource:
6sequential_15_conv2d_47_conv2d_readvariableop_resource;
7sequential_15_conv2d_47_biasadd_readvariableop_resource9
5sequential_15_dense_59_matmul_readvariableop_resource:
6sequential_15_dense_59_biasadd_readvariableop_resource9
5sequential_15_dense_60_matmul_readvariableop_resource:
6sequential_15_dense_60_biasadd_readvariableop_resource9
5sequential_15_dense_61_matmul_readvariableop_resource:
6sequential_15_dense_61_biasadd_readvariableop_resource
identityн
-sequential_15/conv2d_45/Conv2D/ReadVariableOpReadVariableOp6sequential_15_conv2d_45_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-sequential_15/conv2d_45/Conv2D/ReadVariableOpѕ
sequential_15/conv2d_45/Conv2DConv2Dconv2d_45_input5sequential_15/conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..*
paddingVALID*
strides
2 
sequential_15/conv2d_45/Conv2Dд
.sequential_15/conv2d_45/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_conv2d_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_15/conv2d_45/BiasAdd/ReadVariableOpш
sequential_15/conv2d_45/BiasAddBiasAdd'sequential_15/conv2d_45/Conv2D:output:06sequential_15/conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..2!
sequential_15/conv2d_45/BiasAddЈ
sequential_15/conv2d_45/ReluRelu(sequential_15/conv2d_45/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ..2
sequential_15/conv2d_45/Reluн
-sequential_15/conv2d_46/Conv2D/ReadVariableOpReadVariableOp6sequential_15_conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-sequential_15/conv2d_46/Conv2D/ReadVariableOp
sequential_15/conv2d_46/Conv2DConv2D*sequential_15/conv2d_45/Relu:activations:05sequential_15/conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ,,*
paddingVALID*
strides
2 
sequential_15/conv2d_46/Conv2Dд
.sequential_15/conv2d_46/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_conv2d_46_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_15/conv2d_46/BiasAdd/ReadVariableOpш
sequential_15/conv2d_46/BiasAddBiasAdd'sequential_15/conv2d_46/Conv2D:output:06sequential_15/conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ,,2!
sequential_15/conv2d_46/BiasAddЈ
sequential_15/conv2d_46/ReluRelu(sequential_15/conv2d_46/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ,,2
sequential_15/conv2d_46/Reluє
&sequential_15/max_pooling2d_41/MaxPoolMaxPool*sequential_15/conv2d_46/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2(
&sequential_15/max_pooling2d_41/MaxPoolН
!sequential_15/dropout_27/IdentityIdentity/sequential_15/max_pooling2d_41/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2#
!sequential_15/dropout_27/Identityн
-sequential_15/conv2d_47/Conv2D/ReadVariableOpReadVariableOp6sequential_15_conv2d_47_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_15/conv2d_47/Conv2D/ReadVariableOp
sequential_15/conv2d_47/Conv2DConv2D*sequential_15/dropout_27/Identity:output:05sequential_15/conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2 
sequential_15/conv2d_47/Conv2Dд
.sequential_15/conv2d_47/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_conv2d_47_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_15/conv2d_47/BiasAdd/ReadVariableOpш
sequential_15/conv2d_47/BiasAddBiasAdd'sequential_15/conv2d_47/Conv2D:output:06sequential_15/conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2!
sequential_15/conv2d_47/BiasAddЈ
sequential_15/conv2d_47/ReluRelu(sequential_15/conv2d_47/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
sequential_15/conv2d_47/Reluє
&sequential_15/max_pooling2d_42/MaxPoolMaxPool*sequential_15/conv2d_47/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ

 *
ksize
*
paddingVALID*
strides
2(
&sequential_15/max_pooling2d_42/MaxPoolН
!sequential_15/dropout_28/IdentityIdentity/sequential_15/max_pooling2d_42/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2#
!sequential_15/dropout_28/Identity
sequential_15/flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2 
sequential_15/flatten_15/Constз
 sequential_15/flatten_15/ReshapeReshape*sequential_15/dropout_28/Identity:output:0'sequential_15/flatten_15/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2"
 sequential_15/flatten_15/Reshapeг
,sequential_15/dense_59/MatMul/ReadVariableOpReadVariableOp5sequential_15_dense_59_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02.
,sequential_15/dense_59/MatMul/ReadVariableOpл
sequential_15/dense_59/MatMulMatMul)sequential_15/flatten_15/Reshape:output:04sequential_15/dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_15/dense_59/MatMulб
-sequential_15/dense_59/BiasAdd/ReadVariableOpReadVariableOp6sequential_15_dense_59_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_15/dense_59/BiasAdd/ReadVariableOpн
sequential_15/dense_59/BiasAddBiasAdd'sequential_15/dense_59/MatMul:product:05sequential_15/dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
sequential_15/dense_59/BiasAdd
sequential_15/dense_59/ReluRelu'sequential_15/dense_59/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_15/dense_59/Reluв
,sequential_15/dense_60/MatMul/ReadVariableOpReadVariableOp5sequential_15_dense_60_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02.
,sequential_15/dense_60/MatMul/ReadVariableOpл
sequential_15/dense_60/MatMulMatMul)sequential_15/dense_59/Relu:activations:04sequential_15/dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
sequential_15/dense_60/MatMulб
-sequential_15/dense_60/BiasAdd/ReadVariableOpReadVariableOp6sequential_15_dense_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_15/dense_60/BiasAdd/ReadVariableOpн
sequential_15/dense_60/BiasAddBiasAdd'sequential_15/dense_60/MatMul:product:05sequential_15/dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2 
sequential_15/dense_60/BiasAdd
sequential_15/dense_60/ReluRelu'sequential_15/dense_60/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
sequential_15/dense_60/Reluв
,sequential_15/dense_61/MatMul/ReadVariableOpReadVariableOp5sequential_15_dense_61_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,sequential_15/dense_61/MatMul/ReadVariableOpл
sequential_15/dense_61/MatMulMatMul)sequential_15/dense_60/Relu:activations:04sequential_15/dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_15/dense_61/MatMulб
-sequential_15/dense_61/BiasAdd/ReadVariableOpReadVariableOp6sequential_15_dense_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_15/dense_61/BiasAdd/ReadVariableOpн
sequential_15/dense_61/BiasAddBiasAdd'sequential_15/dense_61/MatMul:product:05sequential_15/dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
sequential_15/dense_61/BiasAddІ
sequential_15/dense_61/SoftmaxSoftmax'sequential_15/dense_61/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
sequential_15/dense_61/Softmax|
IdentityIdentity(sequential_15/dense_61/Softmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00:::::::::::::` \
/
_output_shapes
:џџџџџџџџџ00
)
_user_specified_nameconv2d_45_input
	
­
E__inference_conv2d_46_layer_call_and_return_conditional_losses_864269

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ,,*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ,,2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ,,2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ,,2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ..:::W S
/
_output_shapes
:џџџџџџџџџ..
 
_user_specified_nameinputs


)__inference_dense_61_layer_call_fn_864448

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
О
b
F__inference_flatten_15_layer_call_and_return_conditional_losses_864376

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ

 :W S
/
_output_shapes
:џџџџџџџџџ

 
 
_user_specified_nameinputs
щ
d
F__inference_dropout_28_layer_call_and_return_conditional_losses_864353

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ

 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ

 :W S
/
_output_shapes
:џџџџџџџџџ

 
 
_user_specified_nameinputs
Щ6
ў
.__inference_sequential_15_layer_call_fn_864197
conv2d_45_input,
(conv2d_45_conv2d_readvariableop_resource-
)conv2d_45_biasadd_readvariableop_resource,
(conv2d_46_conv2d_readvariableop_resource-
)conv2d_46_biasadd_readvariableop_resource,
(conv2d_47_conv2d_readvariableop_resource-
)conv2d_47_biasadd_readvariableop_resource+
'dense_59_matmul_readvariableop_resource,
(dense_59_biasadd_readvariableop_resource+
'dense_60_matmul_readvariableop_resource,
(dense_60_biasadd_readvariableop_resource+
'dense_61_matmul_readvariableop_resource,
(dense_61_biasadd_readvariableop_resource
identityГ
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_45/Conv2D/ReadVariableOpЫ
conv2d_45/Conv2DConv2Dconv2d_45_input'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..*
paddingVALID*
strides
2
conv2d_45/Conv2DЊ
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_45/BiasAdd/ReadVariableOpА
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..2
conv2d_45/BiasAdd~
conv2d_45/ReluReluconv2d_45/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ..2
conv2d_45/ReluГ
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_46/Conv2D/ReadVariableOpи
conv2d_46/Conv2DConv2Dconv2d_45/Relu:activations:0'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ,,*
paddingVALID*
strides
2
conv2d_46/Conv2DЊ
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_46/BiasAdd/ReadVariableOpА
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ,,2
conv2d_46/BiasAdd~
conv2d_46/ReluReluconv2d_46/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ,,2
conv2d_46/ReluЪ
max_pooling2d_41/MaxPoolMaxPoolconv2d_46/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_41/MaxPool
dropout_27/IdentityIdentity!max_pooling2d_41/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout_27/IdentityГ
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_47/Conv2D/ReadVariableOpи
conv2d_47/Conv2DConv2Ddropout_27/Identity:output:0'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv2d_47/Conv2DЊ
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_47/BiasAdd/ReadVariableOpА
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d_47/BiasAdd~
conv2d_47/ReluReluconv2d_47/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d_47/ReluЪ
max_pooling2d_42/MaxPoolMaxPoolconv2d_47/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ

 *
ksize
*
paddingVALID*
strides
2
max_pooling2d_42/MaxPool
dropout_28/IdentityIdentity!max_pooling2d_42/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2
dropout_28/Identityu
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
flatten_15/Const
flatten_15/ReshapeReshapedropout_28/Identity:output:0flatten_15/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten_15/ReshapeЉ
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02 
dense_59/MatMul/ReadVariableOpЃ
dense_59/MatMulMatMulflatten_15/Reshape:output:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_59/MatMulЇ
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_59/BiasAdd/ReadVariableOpЅ
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_59/BiasAdds
dense_59/ReluReludense_59/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_59/ReluЈ
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02 
dense_60/MatMul/ReadVariableOpЃ
dense_60/MatMulMatMuldense_59/Relu:activations:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_60/MatMulЇ
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_60/BiasAdd/ReadVariableOpЅ
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_60/BiasAdds
dense_60/ReluReludense_60/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_60/ReluЈ
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_61/MatMul/ReadVariableOpЃ
dense_61/MatMulMatMuldense_60/Relu:activations:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_61/MatMulЇ
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_61/BiasAdd/ReadVariableOpЅ
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_61/BiasAdd|
dense_61/SoftmaxSoftmaxdense_61/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_61/Softmaxn
IdentityIdentitydense_61/Softmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00:::::::::::::` \
/
_output_shapes
:џџџџџџџџџ00
)
_user_specified_nameconv2d_45_input
Љ
J
+__inference_dropout_27_layer_call_fn_864309

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
№

*__inference_conv2d_46_layer_call_fn_864280

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ,,*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ,,2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ,,2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ,,2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ..:::W S
/
_output_shapes
:џџџџџџџџџ..
 
_user_specified_nameinputs
Б
Ќ
D__inference_dense_61_layer_call_and_return_conditional_losses_864437

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
№

*__inference_conv2d_47_layer_call_fn_864336

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ
Ќ
D__inference_dense_60_layer_call_and_return_conditional_losses_864415

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :::O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ю
I
+__inference_dropout_27_layer_call_fn_864314

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_47_layer_call_and_return_conditional_losses_864325

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
АJ

I__inference_sequential_15_layer_call_and_return_conditional_losses_864026
conv2d_45_input,
(conv2d_45_conv2d_readvariableop_resource-
)conv2d_45_biasadd_readvariableop_resource,
(conv2d_46_conv2d_readvariableop_resource-
)conv2d_46_biasadd_readvariableop_resource,
(conv2d_47_conv2d_readvariableop_resource-
)conv2d_47_biasadd_readvariableop_resource+
'dense_59_matmul_readvariableop_resource,
(dense_59_biasadd_readvariableop_resource+
'dense_60_matmul_readvariableop_resource,
(dense_60_biasadd_readvariableop_resource+
'dense_61_matmul_readvariableop_resource,
(dense_61_biasadd_readvariableop_resource
identityГ
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_45/Conv2D/ReadVariableOpЫ
conv2d_45/Conv2DConv2Dconv2d_45_input'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..*
paddingVALID*
strides
2
conv2d_45/Conv2DЊ
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_45/BiasAdd/ReadVariableOpА
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..2
conv2d_45/BiasAdd~
conv2d_45/ReluReluconv2d_45/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ..2
conv2d_45/ReluГ
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_46/Conv2D/ReadVariableOpи
conv2d_46/Conv2DConv2Dconv2d_45/Relu:activations:0'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ,,*
paddingVALID*
strides
2
conv2d_46/Conv2DЊ
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_46/BiasAdd/ReadVariableOpА
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ,,2
conv2d_46/BiasAdd~
conv2d_46/ReluReluconv2d_46/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ,,2
conv2d_46/ReluЪ
max_pooling2d_41/MaxPoolMaxPoolconv2d_46/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_41/MaxPooly
dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2
dropout_27/dropout/ConstЗ
dropout_27/dropout/MulMul!max_pooling2d_41/MaxPool:output:0!dropout_27/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout_27/dropout/Mul
dropout_27/dropout/ShapeShape!max_pooling2d_41/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_27/dropout/Shapeн
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype021
/dropout_27/dropout/random_uniform/RandomUniform
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2#
!dropout_27/dropout/GreaterEqual/yђ
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2!
dropout_27/dropout/GreaterEqualЈ
dropout_27/dropout/CastCast#dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2
dropout_27/dropout/CastЎ
dropout_27/dropout/Mul_1Muldropout_27/dropout/Mul:z:0dropout_27/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout_27/dropout/Mul_1Г
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_47/Conv2D/ReadVariableOpи
conv2d_47/Conv2DConv2Ddropout_27/dropout/Mul_1:z:0'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv2d_47/Conv2DЊ
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_47/BiasAdd/ReadVariableOpА
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d_47/BiasAdd~
conv2d_47/ReluReluconv2d_47/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d_47/ReluЪ
max_pooling2d_42/MaxPoolMaxPoolconv2d_47/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ

 *
ksize
*
paddingVALID*
strides
2
max_pooling2d_42/MaxPooly
dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2
dropout_28/dropout/ConstЗ
dropout_28/dropout/MulMul!max_pooling2d_42/MaxPool:output:0!dropout_28/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2
dropout_28/dropout/Mul
dropout_28/dropout/ShapeShape!max_pooling2d_42/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_28/dropout/Shapeн
/dropout_28/dropout/random_uniform/RandomUniformRandomUniform!dropout_28/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

 *
dtype021
/dropout_28/dropout/random_uniform/RandomUniform
!dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2#
!dropout_28/dropout/GreaterEqual/yђ
dropout_28/dropout/GreaterEqualGreaterEqual8dropout_28/dropout/random_uniform/RandomUniform:output:0*dropout_28/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2!
dropout_28/dropout/GreaterEqualЈ
dropout_28/dropout/CastCast#dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ

 2
dropout_28/dropout/CastЎ
dropout_28/dropout/Mul_1Muldropout_28/dropout/Mul:z:0dropout_28/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2
dropout_28/dropout/Mul_1u
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
flatten_15/Const
flatten_15/ReshapeReshapedropout_28/dropout/Mul_1:z:0flatten_15/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten_15/ReshapeЉ
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02 
dense_59/MatMul/ReadVariableOpЃ
dense_59/MatMulMatMulflatten_15/Reshape:output:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_59/MatMulЇ
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_59/BiasAdd/ReadVariableOpЅ
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_59/BiasAdds
dense_59/ReluReludense_59/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_59/ReluЈ
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02 
dense_60/MatMul/ReadVariableOpЃ
dense_60/MatMulMatMuldense_59/Relu:activations:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_60/MatMulЇ
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_60/BiasAdd/ReadVariableOpЅ
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_60/BiasAdds
dense_60/ReluReludense_60/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_60/ReluЈ
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_61/MatMul/ReadVariableOpЃ
dense_61/MatMulMatMuldense_60/Relu:activations:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_61/MatMulЇ
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_61/BiasAdd/ReadVariableOpЅ
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_61/BiasAdd|
dense_61/SoftmaxSoftmaxdense_61/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_61/Softmaxn
IdentityIdentitydense_61/Softmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00:::::::::::::` \
/
_output_shapes
:џџџџџџџџџ00
)
_user_specified_nameconv2d_45_input
Ф
e
F__inference_dropout_28_layer_call_and_return_conditional_losses_864348

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

 *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ

 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ

 :W S
/
_output_shapes
:џџџџџџџџџ

 
 
_user_specified_nameinputs
	
­
E__inference_conv2d_45_layer_call_and_return_conditional_losses_864247

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ..2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ..2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ00:::W S
/
_output_shapes
:џџџџџџџџџ00
 
_user_specified_nameinputs


)__inference_dense_60_layer_call_fn_864426

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :::O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
J
ў
.__inference_sequential_15_layer_call_fn_864145
conv2d_45_input,
(conv2d_45_conv2d_readvariableop_resource-
)conv2d_45_biasadd_readvariableop_resource,
(conv2d_46_conv2d_readvariableop_resource-
)conv2d_46_biasadd_readvariableop_resource,
(conv2d_47_conv2d_readvariableop_resource-
)conv2d_47_biasadd_readvariableop_resource+
'dense_59_matmul_readvariableop_resource,
(dense_59_biasadd_readvariableop_resource+
'dense_60_matmul_readvariableop_resource,
(dense_60_biasadd_readvariableop_resource+
'dense_61_matmul_readvariableop_resource,
(dense_61_biasadd_readvariableop_resource
identityГ
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_45/Conv2D/ReadVariableOpЫ
conv2d_45/Conv2DConv2Dconv2d_45_input'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..*
paddingVALID*
strides
2
conv2d_45/Conv2DЊ
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_45/BiasAdd/ReadVariableOpА
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ..2
conv2d_45/BiasAdd~
conv2d_45/ReluReluconv2d_45/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ..2
conv2d_45/ReluГ
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_46/Conv2D/ReadVariableOpи
conv2d_46/Conv2DConv2Dconv2d_45/Relu:activations:0'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ,,*
paddingVALID*
strides
2
conv2d_46/Conv2DЊ
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_46/BiasAdd/ReadVariableOpА
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ,,2
conv2d_46/BiasAdd~
conv2d_46/ReluReluconv2d_46/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ,,2
conv2d_46/ReluЪ
max_pooling2d_41/MaxPoolMaxPoolconv2d_46/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_41/MaxPooly
dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2
dropout_27/dropout/ConstЗ
dropout_27/dropout/MulMul!max_pooling2d_41/MaxPool:output:0!dropout_27/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout_27/dropout/Mul
dropout_27/dropout/ShapeShape!max_pooling2d_41/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_27/dropout/Shapeн
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype021
/dropout_27/dropout/random_uniform/RandomUniform
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2#
!dropout_27/dropout/GreaterEqual/yђ
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2!
dropout_27/dropout/GreaterEqualЈ
dropout_27/dropout/CastCast#dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2
dropout_27/dropout/CastЎ
dropout_27/dropout/Mul_1Muldropout_27/dropout/Mul:z:0dropout_27/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout_27/dropout/Mul_1Г
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_47/Conv2D/ReadVariableOpи
conv2d_47/Conv2DConv2Ddropout_27/dropout/Mul_1:z:0'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv2d_47/Conv2DЊ
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_47/BiasAdd/ReadVariableOpА
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d_47/BiasAdd~
conv2d_47/ReluReluconv2d_47/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d_47/ReluЪ
max_pooling2d_42/MaxPoolMaxPoolconv2d_47/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ

 *
ksize
*
paddingVALID*
strides
2
max_pooling2d_42/MaxPooly
dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2
dropout_28/dropout/ConstЗ
dropout_28/dropout/MulMul!max_pooling2d_42/MaxPool:output:0!dropout_28/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2
dropout_28/dropout/Mul
dropout_28/dropout/ShapeShape!max_pooling2d_42/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_28/dropout/Shapeн
/dropout_28/dropout/random_uniform/RandomUniformRandomUniform!dropout_28/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

 *
dtype021
/dropout_28/dropout/random_uniform/RandomUniform
!dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2#
!dropout_28/dropout/GreaterEqual/yђ
dropout_28/dropout/GreaterEqualGreaterEqual8dropout_28/dropout/random_uniform/RandomUniform:output:0*dropout_28/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2!
dropout_28/dropout/GreaterEqualЈ
dropout_28/dropout/CastCast#dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ

 2
dropout_28/dropout/CastЎ
dropout_28/dropout/Mul_1Muldropout_28/dropout/Mul:z:0dropout_28/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2
dropout_28/dropout/Mul_1u
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
flatten_15/Const
flatten_15/ReshapeReshapedropout_28/dropout/Mul_1:z:0flatten_15/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten_15/ReshapeЉ
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02 
dense_59/MatMul/ReadVariableOpЃ
dense_59/MatMulMatMulflatten_15/Reshape:output:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_59/MatMulЇ
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_59/BiasAdd/ReadVariableOpЅ
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_59/BiasAdds
dense_59/ReluReludense_59/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_59/ReluЈ
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02 
dense_60/MatMul/ReadVariableOpЃ
dense_60/MatMulMatMuldense_59/Relu:activations:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_60/MatMulЇ
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_60/BiasAdd/ReadVariableOpЅ
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_60/BiasAdds
dense_60/ReluReludense_60/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_60/ReluЈ
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_61/MatMul/ReadVariableOpЃ
dense_61/MatMulMatMuldense_60/Relu:activations:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_61/MatMulЇ
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_61/BiasAdd/ReadVariableOpЅ
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_61/BiasAdd|
dense_61/SoftmaxSoftmaxdense_61/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_61/Softmaxn
IdentityIdentitydense_61/Softmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ00:::::::::::::` \
/
_output_shapes
:џџџџџџџџџ00
)
_user_specified_nameconv2d_45_input
щ
d
F__inference_dropout_27_layer_call_and_return_conditional_losses_864297

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


)__inference_dense_59_layer_call_fn_864404

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч
M
1__inference_max_pooling2d_42_layer_call_fn_863729

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ф
e
F__inference_dropout_27_layer_call_and_return_conditional_losses_864292

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч
M
1__inference_max_pooling2d_41_layer_call_fn_863717

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ю
I
+__inference_dropout_28_layer_call_fn_864370

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ

 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ

 :W S
/
_output_shapes
:џџџџџџџџџ

 
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_863711

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_863723

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ѓ
G
+__inference_flatten_15_layer_call_fn_864382

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ

 :W S
/
_output_shapes
:џџџџџџџџџ

 
 
_user_specified_nameinputs"И-
saver_filename:0
Identity:0Identity_468"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*У
serving_defaultЏ
S
conv2d_45_input@
!serving_default_conv2d_45_input:0џџџџџџџџџ00<
dense_610
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ЕЯ
­S
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+Ў&call_and_return_all_conditional_losses
Џ__call__
А_default_save_signature"ГO
_tf_keras_sequentialO{"class_name": "Sequential", "name": "sequential_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_15", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_45_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_45", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_46", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_27", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_47", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_42", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_28", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_59", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_60", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_61", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_15", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_45_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_45", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_46", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_27", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_47", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_42", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_28", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_59", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_60", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_61", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.00010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ѕ


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"Ю	
_tf_keras_layerД	{"class_name": "Conv2D", "name": "conv2d_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_45", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 1]}}
ѕ	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"Ю
_tf_keras_layerД{"class_name": "Conv2D", "name": "conv2d_46", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_46", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 46, 46, 8]}}

	variables
regularization_losses
 trainable_variables
!	keras_api
+Е&call_and_return_all_conditional_losses
Ж__call__"ђ
_tf_keras_layerи{"class_name": "MaxPooling2D", "name": "max_pooling2d_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ъ
"	variables
#regularization_losses
$trainable_variables
%	keras_api
+З&call_and_return_all_conditional_losses
И__call__"й
_tf_keras_layerП{"class_name": "Dropout", "name": "dropout_27", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_27", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
ї	

&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"а
_tf_keras_layerЖ{"class_name": "Conv2D", "name": "conv2d_47", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_47", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 22, 16]}}

,	variables
-regularization_losses
.trainable_variables
/	keras_api
+Л&call_and_return_all_conditional_losses
М__call__"ђ
_tf_keras_layerи{"class_name": "MaxPooling2D", "name": "max_pooling2d_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_42", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ъ
0	variables
1regularization_losses
2trainable_variables
3	keras_api
+Н&call_and_return_all_conditional_losses
О__call__"й
_tf_keras_layerП{"class_name": "Dropout", "name": "dropout_28", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_28", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
ъ
4	variables
5regularization_losses
6trainable_variables
7	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"й
_tf_keras_layerП{"class_name": "Flatten", "name": "flatten_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ј

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
+С&call_and_return_all_conditional_losses
Т__call__"б
_tf_keras_layerЗ{"class_name": "Dense", "name": "dense_59", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_59", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3200]}}
є

>kernel
?bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"Э
_tf_keras_layerГ{"class_name": "Dense", "name": "dense_60", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_60", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
і

Dkernel
Ebias
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"Я
_tf_keras_layerЕ{"class_name": "Dense", "name": "dense_61", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_61", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
У
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratemmmm&m'm8m9m>m?mDm EmЁvЂvЃvЄvЅ&vІ'vЇ8vЈ9vЉ>vЊ?vЋDvЌEv­"
	optimizer
v
0
1
2
3
&4
'5
86
97
>8
?9
D10
E11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
&4
'5
86
97
>8
?9
D10
E11"
trackable_list_wrapper
Ю

Olayers
Pmetrics
	variables
Qlayer_metrics
regularization_losses
Rlayer_regularization_losses
Snon_trainable_variables
trainable_variables
Џ__call__
А_default_save_signature
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
-
Чserving_default"
signature_map
*:(2conv2d_45/kernel
:2conv2d_45/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А

Tlayers
Umetrics
	variables
Vlayer_metrics
regularization_losses
Wlayer_regularization_losses
Xnon_trainable_variables
trainable_variables
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_46/kernel
:2conv2d_46/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А

Ylayers
Zmetrics
	variables
[layer_metrics
regularization_losses
\layer_regularization_losses
]non_trainable_variables
trainable_variables
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А

^layers
_metrics
	variables
`layer_metrics
regularization_losses
alayer_regularization_losses
bnon_trainable_variables
 trainable_variables
Ж__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А

clayers
dmetrics
"	variables
elayer_metrics
#regularization_losses
flayer_regularization_losses
gnon_trainable_variables
$trainable_variables
И__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_47/kernel
: 2conv2d_47/bias
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
А

hlayers
imetrics
(	variables
jlayer_metrics
)regularization_losses
klayer_regularization_losses
lnon_trainable_variables
*trainable_variables
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А

mlayers
nmetrics
,	variables
olayer_metrics
-regularization_losses
player_regularization_losses
qnon_trainable_variables
.trainable_variables
М__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А

rlayers
smetrics
0	variables
tlayer_metrics
1regularization_losses
ulayer_regularization_losses
vnon_trainable_variables
2trainable_variables
О__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А

wlayers
xmetrics
4	variables
ylayer_metrics
5regularization_losses
zlayer_regularization_losses
{non_trainable_variables
6trainable_variables
Р__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
": 	 2dense_59/kernel
: 2dense_59/bias
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
Б

|layers
}metrics
:	variables
~layer_metrics
;regularization_losses
layer_regularization_losses
non_trainable_variables
<trainable_variables
Т__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
!: @2dense_60/kernel
:@2dense_60/bias
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
Е
layers
metrics
@	variables
layer_metrics
Aregularization_losses
 layer_regularization_losses
non_trainable_variables
Btrainable_variables
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_61/kernel
:2dense_61/bias
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
Е
layers
metrics
F	variables
layer_metrics
Gregularization_losses
 layer_regularization_losses
non_trainable_variables
Htrainable_variables
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
П

total

count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


total

count

_fn_kwargs
	variables
	keras_api"П
_tf_keras_metricЄ{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
/:-2Adam/conv2d_45/kernel/m
!:2Adam/conv2d_45/bias/m
/:-2Adam/conv2d_46/kernel/m
!:2Adam/conv2d_46/bias/m
/:- 2Adam/conv2d_47/kernel/m
!: 2Adam/conv2d_47/bias/m
':%	 2Adam/dense_59/kernel/m
 : 2Adam/dense_59/bias/m
&:$ @2Adam/dense_60/kernel/m
 :@2Adam/dense_60/bias/m
&:$@2Adam/dense_61/kernel/m
 :2Adam/dense_61/bias/m
/:-2Adam/conv2d_45/kernel/v
!:2Adam/conv2d_45/bias/v
/:-2Adam/conv2d_46/kernel/v
!:2Adam/conv2d_46/bias/v
/:- 2Adam/conv2d_47/kernel/v
!: 2Adam/conv2d_47/bias/v
':%	 2Adam/dense_59/kernel/v
 : 2Adam/dense_59/bias/v
&:$ @2Adam/dense_60/kernel/v
 :@2Adam/dense_60/bias/v
&:$@2Adam/dense_61/kernel/v
 :2Adam/dense_61/bias/v
м2й
I__inference_sequential_15_layer_call_and_return_conditional_losses_864026
I__inference_sequential_15_layer_call_and_return_conditional_losses_864078Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
І2Ѓ
.__inference_sequential_15_layer_call_fn_864197
.__inference_sequential_15_layer_call_fn_864145Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
я2ь
!__inference__wrapped_model_863705Ц
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *6Ђ3
1.
conv2d_45_inputџџџџџџџџџ00
я2ь
E__inference_conv2d_45_layer_call_and_return_conditional_losses_864247Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv2d_45_layer_call_fn_864258Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv2d_46_layer_call_and_return_conditional_losses_864269Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv2d_46_layer_call_fn_864280Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Д2Б
L__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_863711р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
1__inference_max_pooling2d_41_layer_call_fn_863717р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ъ2Ч
F__inference_dropout_27_layer_call_and_return_conditional_losses_864297
F__inference_dropout_27_layer_call_and_return_conditional_losses_864292Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
+__inference_dropout_27_layer_call_fn_864314
+__inference_dropout_27_layer_call_fn_864309Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
я2ь
E__inference_conv2d_47_layer_call_and_return_conditional_losses_864325Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv2d_47_layer_call_fn_864336Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Д2Б
L__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_863723р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
1__inference_max_pooling2d_42_layer_call_fn_863729р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ъ2Ч
F__inference_dropout_28_layer_call_and_return_conditional_losses_864348
F__inference_dropout_28_layer_call_and_return_conditional_losses_864353Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
+__inference_dropout_28_layer_call_fn_864365
+__inference_dropout_28_layer_call_fn_864370Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
№2э
F__inference_flatten_15_layer_call_and_return_conditional_losses_864376Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_flatten_15_layer_call_fn_864382Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_dense_59_layer_call_and_return_conditional_losses_864393Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_dense_59_layer_call_fn_864404Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_dense_60_layer_call_and_return_conditional_losses_864415Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_dense_60_layer_call_fn_864426Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_dense_61_layer_call_and_return_conditional_losses_864437Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_dense_61_layer_call_fn_864448Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
;B9
$__inference_signature_wrapper_864236conv2d_45_inputЋ
!__inference__wrapped_model_863705&'89>?DE@Ђ=
6Ђ3
1.
conv2d_45_inputџџџџџџџџџ00
Њ "3Њ0
.
dense_61"
dense_61џџџџџџџџџЕ
E__inference_conv2d_45_layer_call_and_return_conditional_losses_864247l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ00
Њ "-Ђ*
# 
0џџџџџџџџџ..
 
*__inference_conv2d_45_layer_call_fn_864258_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ00
Њ " џџџџџџџџџ..Е
E__inference_conv2d_46_layer_call_and_return_conditional_losses_864269l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ..
Њ "-Ђ*
# 
0џџџџџџџџџ,,
 
*__inference_conv2d_46_layer_call_fn_864280_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ..
Њ " џџџџџџџџџ,,Е
E__inference_conv2d_47_layer_call_and_return_conditional_losses_864325l&'7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
*__inference_conv2d_47_layer_call_fn_864336_&'7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџ Ѕ
D__inference_dense_59_layer_call_and_return_conditional_losses_864393]890Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ 
 }
)__inference_dense_59_layer_call_fn_864404P890Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ Є
D__inference_dense_60_layer_call_and_return_conditional_losses_864415\>?/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ@
 |
)__inference_dense_60_layer_call_fn_864426O>?/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ@Є
D__inference_dense_61_layer_call_and_return_conditional_losses_864437\DE/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ
 |
)__inference_dense_61_layer_call_fn_864448ODE/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџЖ
F__inference_dropout_27_layer_call_and_return_conditional_losses_864292l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ "-Ђ*
# 
0џџџџџџџџџ
 Ж
F__inference_dropout_27_layer_call_and_return_conditional_losses_864297l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ "-Ђ*
# 
0џџџџџџџџџ
 
+__inference_dropout_27_layer_call_fn_864309_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ " џџџџџџџџџ
+__inference_dropout_27_layer_call_fn_864314_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ " џџџџџџџџџЖ
F__inference_dropout_28_layer_call_and_return_conditional_losses_864348l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ

 
p
Њ "-Ђ*
# 
0џџџџџџџџџ

 
 Ж
F__inference_dropout_28_layer_call_and_return_conditional_losses_864353l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ

 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ

 
 
+__inference_dropout_28_layer_call_fn_864365_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ

 
p
Њ " џџџџџџџџџ

 
+__inference_dropout_28_layer_call_fn_864370_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ

 
p 
Њ " џџџџџџџџџ

 Ћ
F__inference_flatten_15_layer_call_and_return_conditional_losses_864376a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ

 
Њ "&Ђ#

0џџџџџџџџџ
 
+__inference_flatten_15_layer_call_fn_864382T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ

 
Њ "џџџџџџџџџя
L__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_863711RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
1__inference_max_pooling2d_41_layer_call_fn_863717RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџя
L__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_863723RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
1__inference_max_pooling2d_42_layer_call_fn_863729RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџЬ
I__inference_sequential_15_layer_call_and_return_conditional_losses_864026&'89>?DEHЂE
>Ђ;
1.
conv2d_45_inputџџџџџџџџџ00
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Ь
I__inference_sequential_15_layer_call_and_return_conditional_losses_864078&'89>?DEHЂE
>Ђ;
1.
conv2d_45_inputџџџџџџџџџ00
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Є
.__inference_sequential_15_layer_call_fn_864145r&'89>?DEHЂE
>Ђ;
1.
conv2d_45_inputџџџџџџџџџ00
p

 
Њ "џџџџџџџџџЄ
.__inference_sequential_15_layer_call_fn_864197r&'89>?DEHЂE
>Ђ;
1.
conv2d_45_inputџџџџџџџџџ00
p 

 
Њ "џџџџџџџџџС
$__inference_signature_wrapper_864236&'89>?DESЂP
Ђ 
IЊF
D
conv2d_45_input1.
conv2d_45_inputџџџџџџџџџ00"3Њ0
.
dense_61"
dense_61џџџџџџџџџ