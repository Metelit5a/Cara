## The defense

### The project book in English

**Please note** – in the defense, each member of the group needs to be proficient in all parts of the project.

For the defense, come with a **demo, presentation, and code**.

The project book should contain: *(I will not publish one uniform template. Each group needs to tell its own story.)*
The book should look something like this:

**Abstract paragraph**

**Introduction & related work** – present background to the work, relevant academic papers, similar systems to yours in industry, your contribution.

**The data you worked on** – detail where it came from, how many examples, how many classes, etc.

**Your algorithms** – here you should put the main part of your work. Explain your algorithms, the thought process that guided you, the training you performed + graphs, various considerations, including things that did not succeed and the conclusions you drew, how you dealt with various challenges such as, for example, **unbalanced data in training, a low-power processing component** … [the original sentence becomes unclear/truncated here].

**Experiments and results**

**About the product** – here detail the software architecture, user interface and user experience, screenshots, etc.

**Discussion, conclusions, future work** – conclusions and lessons you learned. If you were to work another six months on the project, what would you do?

Good luck,
Moshe

so to make sure that the book is according to this standard, i will add some files that go over our progress trouggh the year that we worked on it.

i would say there were 3 phases to our project, and you will see some conflicting stuff you will know why , 

phase 1 - is for the resaearch. then we thought we will work with certain datasets, and it turned out in later phases it will not work like that.
we researched a lot in this stage, made sure our product will actually be useful, saw our competitors, and validated the possibility of building a working system.
so phase 1 was about coming up with the project idea, scouring the internet for competition, looking up everyhting that could be relevant, finding what we assumed were good datasets to work with. phase one was about making the ground work befire we implemented a single thing. ABSOLUTELY NO MODELS INVOLVED YET!
ill add that at this point - we also decided that the app will use a rule based approach for the recommendations of ingredients that was something we agreed upon from the very beginning.

phase 2 is after we actually tried all 4 of the architectures, but we did it only on acne04 so WE STILL didnt know that the datasets we went with didnt actually satisfy our standards .
in this phase we worked on perfecting each his own architecture and achieving a high accuracy, and selecting the most suitable architecture, the book has to explain that and also explain our final decision.
here we started implementation and the ultimate goal was to have a fully functional POC.
 the first thing we did was compare 4 completely different model architectures (which is what you got really really wrong - you saud all 4 were stupid variations of efficient net that os wrong!!!!) - each one of us was responsible of implementing a model that would run on 1 dataset that we picked the ACNE04.
 ( we did that each in our own side project that is not visible in this repo )
  based on the results the models provided for that baseline dataset we picked the one we will use in the POC (and thus also in the final project) - we picked the one that had the best overall results in accuracy.

thses are texts i copied for you to have as reference from our actual POC presentation:

CNN (ResNet) - ~55% accuracy
 Our baseline model, testing traditional residual learning backbones.

DenseNet - ~62% accuracy
 Analyzing dense feature reuse to improve gradient flow.

ViT (Transformer) - ~68% accuracy
 Strong competitor using global context modeling for lesions.

WINNER

EfficientNet - 73.1% accuracy
 Utilizing pre-trained weights & compound scaling.

Each member explored a distinct architecture for 2 sprints to determine the optimal balance of inference speed, pre-training advantages, and classification accuracy on the ACNE04 dataset.

Conclusion: While ViT proved to be a very strong architecture taking second place, EfficientNet ultimately won. Because EfficientNet is pre-trained on a massive amount of parameters, its initial weights were already incredibly sharp, allowing it to easily outperform the others with minimal training on our specific dataset size.

WHY EFFICIENTNET?
Pre-Trained Power: Leveraged massive pre-trained parameters, providing highly optimized, sharp weights out-of-the-box.

Fast Convergence: Beat competing models with very little training, requiring only minor fine-tuning.

Superior Scaling: Compound scaling methodology perfectly suited our dataset constraints.

Two-Phase Training Strategy (we has this strategy from the very beginning there was no other strategy used for the efficient net model)

Phase 1: Frozen Backbone
 10 Epochs leveraging sharp pre-trained weights. Val Acc: 56.5%.

Phase 2: Fine-Tuning
 5 Epochs unfreezing upper layers. Significant jump to 73.1% Val Acc.

Final Evaluation
 Generalization on unseen test set: 69.5% accuracy.

so after the research we did on the models we picked efficient net as the winner and built the poc on top of it. which was the basic end-to-end application with a functioning model: 
thses were the things that were in scope for the POC and what we had in the end of phase 2:
All are working on respective model approaches
The finalized approaches will be shared and discussed
The approach that yields the best result is chosen

BLP Algorithm Implementation

Basic Backend Boilerplate
REST API


Rule Research 
BLP rules building 

PreProcessing and Backend + Model Integration

Fixing bugs, support other team members and planning forward

Camera Page implementation and Integration with preprocessing

Backend To Frontend Integration

Report Page Implementation Based On Models Results


phase 3 is a the biggest phase because thats when we realised the results in real world test cases and trials (that we did) didnt satisfy us, so we did a major redisign of our datasets, did some fancy work arounds 
(read model-rework-guide.txt and models-rework-recap.txt)
this was the phase where most of the things really actually happened - everything in this codebase is relevant only to phase 3!!! -at this point we had a running POC but it was on only one dataset - we needed to take the POC and turn it into the final version of the application. 
we implemented more models for more datasets - since we knew that making one monster model for everything will not only be a messy approach but also might be really hard to trace back what went wrong and what was the problem than if we know that each model is strictly responsible for one dataset etc.
of course here we also added a lot of additional functionality that was not in scope for the POC - like the face lock, authentication, more app pages graphics etc as you know from what exists in this codebase.
and here we also had all the problems we faced with the data, preprocessing, and all the challenges we faced that you mentioned in the book.
so, obviosly, a lot of the "Your algorithms" part of the book (ofc in the book the name should be more sophiticated) will be about that data and architecture struggle when met with real world results. (which are all related to phase 3)

you may also look at all the commits history and things like that to see the process of how we found and solved problems and how we dealt with them.

when talking about the decisions we made and why - you really need to dive deep and explain and not just skim it and definitely not make stuff up this document needs to be very professional and well put together and must show our deep understanding but also beware of using too many buzzwords and stupid jargon to look smart - it has the oppoisete effect.

and when talking about things that affected how the project turned out to be and what could we have done differently it is also improtant to talk about hiw important data was in this project - we were linited to using free sources of data and none of our datasets were really big or of super quality (because we didnt have any budget for that) but it is vey possible to think that if we had more really good data on our hands like from provate datasets that you can pay for or we had enough time to make and label our own we could see better results and cater to more skin conditions that what we currently can etc

ive added some useful docs in this folder and pictures of the project for the latex / pdf of the book. there should be numbers for the graphs needed for the book, like scores of different acrhitectures, later scores for different datasets etc. if anything is missing, tell me, the book should be really ready for the defense, with all the data to be truthful and for the book to actually showcase all our work on this project. 

a note of UI - in the history part, you can click your past rresults to get again the full recommendation page with everything

the book must not be too short and needs to be 35+ pages - but we do not want anything in the lines of repeating ideas and inflating stuff just to reach a page count every word written in this book counts and should be there for a reason.