import { Component, Input } from '@angular/core';

@Component({
  selector: 'mc-genres-chart',
  templateUrl: 'chart.component.html',
  styleUrls: [ 'chart.component.css' ]
})
export class GenresChart {

  @Input()
  results: any[];

  /**
   * The size of the chart
   */
  view = [550, 400]

  /**
   * Show labels with arrows pointing to each genre in the chart
   */
  showLabels = true;

  /**
   * Doughnut style chart
   */
  doughnut = true;

  constructor() {
  }
  
  /**
   * Create select function so ngxchart does not break on calling a undefined function.
   */
  onSelect(_) { }
  
}