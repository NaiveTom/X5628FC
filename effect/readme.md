Here are some training effects and reflections


I adjusted the model to make the model as simple as possible, including reducing the number of CNN layers and reducing the number of nodes in the dense layer and dense layer, so that the entire model can be better trained without various gradient disappearance, gradient dispersion, and The problem of model loss changes too fast and jumps.

If the gradient disappears or jumps, the ROC curve will be displayed like this
![image](https://github.com/NaiveTom/X5628FC/blob/main/effect/The%20AUC%20curve%20collapses%20after%20the%20gradient%20Vanished.PNG)

This picture is the effect after 10,000 clips training
![image](https://github.com/NaiveTom/X5628FC/blob/main/effect/After%20using%20a%20simple%20model%2C%20training%20becomes%20more%20simple%20and%20effective%2C%20and%20the%20gradient%20can%20be%20in%20a%20downward%20direction.PNG)
