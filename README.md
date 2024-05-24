# bachelorProef
## Introductie
In ons hedendaags leven valt het haast niet te ontwijken: Artificiële Intelligentie. Dezer dagen kent deze term vele variaties en toepassingen. Deze evolutie van artificiële intelligentie is sterk gefundeerd op \textit{feed-forward} neurale netwerken. Terwijl deze neurale netwerken al enkele decennia meegaan zijn deze nog altijd sterk van belang in nieuwe vormen van artificiële intelligentie. Ze vormen wel degelijk een basis. In deze paper nemen we deze onder de loep en bekijken hoe goed deze juist zijn met hun benaderende eigenschappen. We bespreken hier ten eerste hoe neurale netwerk eigenlijk in elkaar zitten en kunnen convergeren naar een punt dat optimaal is. We zullen dan een neuraal netwerk zijn potentieel uitgebreid bespreken in de vorm van de Universele Approximatie Stelling. Een stelling die in literatuur veel vormen kent, met elks verschillende voorwaarden en gevolgen. Vele draaien rond abstracte ideeën in de analyse. We zullen zelf proberen een algemene, vatbare versie hier van te creëren dat niet te diep ingaat op abstracte theorie. Dit namelijk op basis van een visueel idee. Hierna zullen we verder gaan naar hyperparameters voor neurale netwerken, een term die de lezer nog zal tegenkomen. Een neuraal netwerk is namelijk een bundel van trainbare parameters en parameters om deze vorige te trainen. Net zoals deze trainbare aangepast kunnen worden naar keuze en dus naar een optimale oplossing, kunnen we ook deze hyperparameters, de parameters van de parameters, instellen zodat alle vorige processen efficiënter verlopen.
##Mnist en MnistBackprop
Deze folder bevat de betanden AI_python en MNIST_load: AI_python is het bestand, gevuld met de gebruikte functies. MNIST_LOAD is het bestand waar de het neuraalnetwerk in getraind wordt. Eerst zal dit programma de mnistdataset inladen. Dit is een dataset die bestaat uit handgetekende cijfers. Hierna zal het netwerk zich trainen op het classificatieprobleem.
Mnistbackprop doet exact hetzelfde maar dan zal deze werken met behulp van backpropagation in plaats van de naïve implementatie.
##ADAM
Dit is een bestand dat de functies bevat om een neuraal netwerk te trainen met behulp van de ADAM optimizer.
##Stochasticdescent
Dit is een bestand dat een neuraal netwerk gaat trainen op een testdataset van sin(x). Dit met behulp van de naive implementatie besproken in de paper.
##tensorflow
Dit is het bestand gebruikt om de experimenten op te stellen die verschillende topologïen uittesten. Deze maakt niet gebruik van het zelfgeprogrameerde netwerk maar van de library tensorflow.
##
