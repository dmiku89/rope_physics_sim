# rope_physics_sim
***
<h2> Физическая модель динамики троса </h2>
Простая модель на python, описывающаю динамическую модель поведения троса с одним закрепленным и с одним свободным концом с грузом. 
Реализация модели основана на курсе лекций
<a href="https://www.youtube.com/watch?v=0WaDxYuD9S8&list=PLdCdV2GBGyXMyvIX7Llch6-Y_7dStrnSr">отсюда</a>.
</br>Трос описывается набором звеньев, для каждого из которых задаются параметры упругости (модуль Юнга), демпфирования (поглощения энергии растяжения), массы, а также величина коэффициента аэродинамического сопротивления. 

</br>Настраиваемые параметры:
  * количество звеньев;
  * масса троса;
  * масса груза (масса на последнем подвижном звене);
  * модуль юнга звена;
  * кэффициент демпфирования звена;
  * начальное положение последнего звена;
  * начальная скорость;
  * шаг интегрирования.
***
