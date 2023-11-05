Metin tabanlı datasetler için xgboost ve nativebayes multinominalNB
-------------------------------------------------------------------------------------------------------------------------------------------------------------

MultinomialNB, Naive Bayes algoritmasının bir türüdür ve özellikle metin sınıflandırma gibi bazı özel uygulamalarda iyi çalışabilir.
Ancak, bu algoritmanın bazı varsayımları vardır, özellikle "naive" varsayım, gerçek dünyadaki karmaşık ilişkileri tam olarak yansıtmayabilir.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

XGBoost ise bir ensemble öğrenme algoritmasıdır ve ağaç tabanlı bir model oluşturur.
Genellikle geniş bir uygulama yelpazesi için iyi performans gösterir ve daha karmaşık ilişkileri yakalayabilir.
Ancak, daha fazla hiperparametre ayarı gerektirebilir ve aşırı uyumu (overfitting) önlemek için dikkatli bir şekilde ayarlanmalıdır.

-------------------------------------------------------------------------------------------------------------------------------------------------------------
Sayısal veriler için Forest, Tree, Logistic modellerini kullanabiliriz
-------------------------------------------------------------------------------------------------------------------------------------------------------------

Forest: "Forest" terimi, genellikle "Random Forest" veya "Gradient Boosting" gibi ensemble (toplu) öğrenme yöntemlerini ifade etmek için kullanılır.
Bu yöntemler, birden çok ağaç veya modeli birleştirerek daha iyi tahminler yapmayı amaçlar.
Random Forest ve XGBoost, ensemble öğrenme algoritmalarının örnekleridir.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

Tree: "Tree" terimi, genellikle "Decision Tree" veya "Regression Tree" gibi ağaç tabanlı öğrenme modellerini ifade etmek için kullanılır.
Karar ağaçları, sınıflandırma veya regresyon problemlerini çözmek için kullanılır.
Bu ağaçlar, veriye göre sıralanmış karar düğümlerinden oluşur ve her düğüm, bir kararın alındığı yerdir.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

Logistic: "Logistic" terimi, genellikle "Logistic Regression" modelini ifade etmek için kullanılır.
Logistic Regression, sınıflandırma problemleri için kullanılan bir istatistiksel modeldir.
Ancak isminde "regresyon" geçmesine rağmen, aslında bir sınıflandırma modelidir ve iki sınıf arasında bir karar sınırı oluşturur.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

metini sayısal verilere çevirmek için "CountVectorizer" kullanarak bunları sayısal verilere ceviririz.
yine metin ama yes no gibi secenekleri varsa "pd.get_dummies" ile ayrı sütun olarak ayırıp 0 1 koyarak sayısal verilere dönüştürürz.
