import { Component } from '@angular/core';

@Component({
  selector: 'mc-classification',
  templateUrl: './classification.component.html',
  styleUrls: [ './classification.component.css' ]
})
export class ClassificationComponent {
  
  private _url : string;

  public get url() : string {
    return this._url;
  }

  public set url(newUrl : string) {
    this._url = newUrl;
  }

  public classify($event: KeyboardEvent) {
    console.log(this);
  }
  
} 