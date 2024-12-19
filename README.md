# Monitoring Dashboard

This Streamlit app at the current stage visualizes different metrics of data drift for deployed machine learning models. The new data with which the model generates new target predictions is compared to the data the model was trained on. The app shows a similar comparison for the actual predictions, relative to the train/test predictions during development. 

The different pages show different parts of the data: whether to compare in/out-of-sample (or implementation, or non-labeled) data, or whether to look at actual predictions. Each page then shows several metrics reflecting statisical differences between data distributions (number of data points, differences in mean and standard deviation, and a Kolmogorov-Smirnov test). Each metric is calculated for each time point at which there was a new dataset fed into the model; these "measurements of difference" are then plotted over time. You can interact with these plots by using the menus in the left sidebar: you can choose different features to look into (only the top-10 most important features are shown) and different time windows. 

This app should be part of maintenance during deployment phase of a model. Look for signs of data and/or prediction drift: is the time series in the plot static, do you see a peak, or a gradual change?. This may signal the need to retrain and/or adapt the model, because it may reflect the fact that the model is "aging" with respect to the current state of the target group it is designed for.

This app should be quite generic for any machine learning model. Just set the correct path for the input data and the prediction data. Also provide a datafile with general feature importance. The app was designed with the ILT Inland Shipping model as test case, which is a random forest model.

## Installation

### Create new Git repository

You should use this repo as a template, but connect it to a new repo specific to your ML monitoring project! To do so, follow these steps:

1. Within GitLab, first create a blank project: Create new project &rarr; Create blank project
2. Choose a project name, choose Internal, uncheck 'Initialize repository with a README' (so it should NOT have a check mark)
3. Click 'Create project'; you will then see an empty repository with Terminal instructions on how to fill it.
5. In JupyterLab, create a new empty folder for your monitoring project, open a new Terminal and navigate to this folder using `cd`.
6. Type the following commands:

```console
$ git init
$ git remote add origin https://gitlab.datascience.rijkscloud.nl/joramvandriel/ai-monitoring-streamlit-dashboard.git
$ git pull origin master
$ git remote rename origin old-origin
$ git remote add origin https://gitlab.datascience.rijkscloud.nl/[your-account]/[your-new-monitoring-project].git
$ git push origin --all
```

### Conda

For the app to work properly, you should create a conda environment and activate it. This repository contains a .yml file to be able to create this environment with the right packages and depdendencies. In a Terminal, run:

`bash setenv.sh`

This will create the conda env `monitoring-dashboard`. To activate it so that you have the right packages, run: 

`conda activate monitoring-dashboard`

You will see that `(base)` changed into `(monitoring-dashboard)` at the far left side of each line in the Terminal.

## Adaptation

Your app won't work right away. It is designed in such a way that you only need to adapt prepareData.py according to the specifics of the data and the model. This has been tested on the Shipbreaking model. With only some minor changes in path names and references to column names and label names, prepareData.py was changed from the Inland Shipping model to the Shipbreaking model and the monitoring streamlit app worked properly. The idea is that in essence you don't need to change the other .py scripts. Of course, each model is different and diverges in the way the data, features etc look like. It could be that adapting prepareData.py involves much more (for example, it currently involves a section with Gini values of global feature importance, which is spefici to tree-based models such as Random Forest). In general, if you have regular updates of new datasets for predictions that come without new labels, and your model is a supervised model with labels on the original dataset the model was trained on, then this app can be used. 

## Usage

If prepareData.py is adapted to your needs, then updating and running the app involves two steps in the Terminal:

```console
$ python prepareData.py
$ streamlit run app.py
```

This will show in the Terminal a port number (usually something like 8502). In your browser, type in https://jupyter.ilt-analyse.rijkscloud.nl/user/[your-account-name]/proxy/[portnumber]/ (don't forget the ending slash).

Update, look at and interpret the monitoring dashboard regularly: for example each time there is a data update for the deployed model (or each month when this is too frequent). Discuss the results with the involved data scientists and summarize your conclusions in the user group meetings.
